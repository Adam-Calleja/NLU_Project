"""

Architecture Overview:
    1. Backbone   : DeBERTa-v3-large for contextual embeddings.
    2. Sequential : 8-head Self-Attention + 2-layer BiLSTM for sequence modeling.
    3. Interaction: Explicit extraction of Claim (u) and Evidence (v).
    4. Features   : Concatenates [CLS, BiLSTM_pooled, |u-v|, u*v].
    5. Classifier : Multi-Sample Dropout + Linear Head (4096 -> 2).

Training Pipeline:
    - Stage 1 (Balanced Setup) : WordNet synonym augmentation to balance classes.
    - Stage 2 (Hard Negatives) : Fine-tuning on tricky examples using dynamic 
                                 class weighting to penalize minority misclassifications.
    - Regularization           : R-Drop Loss (KL Divergence) & Label Smoothing.
    - Optimization             : AdamW with differential learning rates and linear warmup.
"""

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             accuracy_score, matthews_corrcoef,
                             classification_report)
import nlpaug.augmenter.word as naw

# 1. Configuration & Hyperparameters
MODEL_NAME         = "microsoft/deberta-v3-large"
MAX_LEN            = 256
BATCH_SIZE         = 16
ACCUMULATION_STEPS = 4
LEARNING_RATE      = 1e-5
STAGE1_EPOCHS      = 7
STAGE2_EPOCHS      = 2
STAGE2_LR_FACTOR   = 0.2
PATIENCE           = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

# 2. Data Augmentation
def augment_minority_class(df, target_label=1, multiplier=2, seed=42):
    """Augments minority class samples using WordNet synonym replacement."""
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.15)
    minority_df = df[df["label"] == target_label].reset_index(drop=True)

    print(f"[INFO] Augmenting {len(minority_df)} Class {target_label} samples ({multiplier}x)...")
    augmented_rows = []
    
    for idx, (_, row) in enumerate(minority_df.iterrows()):
        for _ in range(multiplier):
            try:
                new_claim    = aug.augment(str(row["Claim"]))[0]
                new_evidence = aug.augment(str(row["Evidence"]))[0]
            except Exception:
                new_claim    = str(row["Claim"])
                new_evidence = str(row["Evidence"])
                
            augmented_rows.append({
                "Claim": new_claim, 
                "Evidence": new_evidence,
                "label": int(target_label),
            })

    aug_df = pd.DataFrame(augmented_rows)
    final_df = pd.concat([df, aug_df], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return final_df

# 3. Custom Loss Function
def rdrop_loss(logits1, logits2, targets, class_weights, alpha=0.5):
    """Computes CrossEntropy + KL divergence between two dropout passes (R-Drop)."""
    ce = nn.CrossEntropyLoss(weight=class_weights.to(logits1.dtype), label_smoothing=0.05)
    ce_loss = 0.5 * (ce(logits1, targets) + ce(logits2, targets))

    p = F.log_softmax(logits1, dim=1)
    q = F.log_softmax(logits2, dim=1)
    kl = 0.5 * (F.kl_div(p, q.exp(), reduction="batchmean") + F.kl_div(q, p.exp(), reduction="batchmean"))
    
    return ce_loss + alpha * kl

# 4. Dataset Loader
class EvidenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data.reset_index(drop=True) if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        self.data = self.data.dropna(subset=["Claim", "Evidence", "label"]).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        enc = self.tokenizer(
            str(row["Claim"]).strip(), str(row["Evidence"]).strip(),
            add_special_tokens=True, max_length=self.max_len,
            padding="max_length", truncation=True,
            return_attention_mask=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "targets": torch.tensor(int(row["label"]), dtype=torch.long),
        }

# 5. Model Architecture
class AdvancedDebertaCrossEncoder(nn.Module):
    def __init__(self, model_name, tokenizer):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        H = self.deberta.config.hidden_size
        self.sep_id = tokenizer.sep_token_id

        self.attention = nn.MultiheadAttention(embed_dim=H, num_heads=8, dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(H)
        self.bilstm    = nn.LSTM(input_size=H, hidden_size=H // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        
        self.dropouts   = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])
        self.layer_norm = nn.LayerNorm(H * 4)
        self.classifier = nn.Linear(H * 4, 2)
        
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        out = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state.to(torch.float32)
        cls_embed = hidden[:, 0, :]

        pad_mask = (attention_mask == 0)
        attn_out, _ = self.attention(hidden, hidden, hidden, key_padding_mask=pad_mask)
        hidden = self.attn_norm(hidden + attn_out)

        lstm_out, _ = self.bilstm(hidden)
        mask_exp = attention_mask.unsqueeze(-1).float()
        lstm_pooled = (lstm_out * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

        B = input_ids.size(0)
        diff_vecs, prod_vecs = [], []
        
        for i in range(B):
            sep_pos = (input_ids[i] == self.sep_id).nonzero(as_tuple=True)[0]
            if len(sep_pos) >= 2:
                s1, s2 = sep_pos[0].item(), sep_pos[1].item()
                c_tok  = hidden[i, 1:s1, :]
                e_tok  = hidden[i, s1+1:s2, :]
            else:
                mid   = hidden.size(1) // 2
                c_tok = hidden[i, 1:mid, :]
                e_tok = hidden[i, mid:, :]
                
            if c_tok.size(0) == 0: c_tok = hidden[i, [0], :]
            if e_tok.size(0) == 0: e_tok = hidden[i, [0], :]
            
            u = F.normalize(c_tok.mean(0), p=2, dim=-1)
            v = F.normalize(e_tok.mean(0), p=2, dim=-1)
            diff_vecs.append((u - v).abs())
            prod_vecs.append(u * v)

        diff = torch.stack(diff_vecs)
        prod = torch.stack(prod_vecs)
        
        feat = self.layer_norm(torch.cat([cls_embed, lstm_pooled, diff, prod], dim=1))
        logits = sum(self.classifier(d(feat)) for d in self.dropouts) / len(self.dropouts)

        return logits, cls_embed

# 6. Evaluation & Optimizers
def evaluate(model, data_loader):
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in data_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            tgt  = batch["targets"].to(DEVICE)
            
            logits, _ = model(ids, mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(tgt.cpu().numpy())
            
    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(true_labels, predictions, average="macro"),
        "mcc":      matthews_corrcoef(true_labels, predictions),
        "report":   classification_report(true_labels, predictions, target_names=["Not Evidence (0)", "Evidence (1)"], zero_division=0),
    }

def build_optimizer(model, lr):
    """Creates an AdamW optimizer with differential learning rates for robust fine-tuning."""
    encoder_params = list(model.deberta.parameters())
    new_module_params = (list(model.bilstm.parameters()) + list(model.attention.parameters()) + 
                         list(model.attn_norm.parameters()) + list(model.layer_norm.parameters()))
    classifier_params = list(model.classifier.parameters())
    
    return torch.optim.AdamW([
        {"params": encoder_params,    "lr": lr,     "weight_decay": 0.01},
        {"params": new_module_params, "lr": lr * 2, "weight_decay": 0.01},
        {"params": classifier_params, "lr": lr * 5, "weight_decay": 0.01},
    ], eps=1e-6)

# 7. Training Loop
def run_epoch(model, loader, optimizer, scheduler, class_weights, epoch_num, total_epochs, stage_label):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(DEVICE, non_blocking=True)
        mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        tgt  = batch["targets"].to(DEVICE, non_blocking=True)

        logits1, _ = model(ids, mask)
        logits2, _ = model(ids, mask)

        loss = rdrop_loss(logits1, logits2, tgt, class_weights, alpha=0.5) / ACCUMULATION_STEPS
        loss.backward()
        total_loss += loss.item() * ACCUMULATION_STEPS

        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == len(loader) - 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"[{stage_label}] Epoch {epoch_num}/{total_epochs} | Step {step+1}/{len(loader)} | Loss: {loss.item()*ACCUMULATION_STEPS:.4f} | LR: {lr:.2e}")

    return total_loss / len(loader)

def train():
    required_files = ["ED/train.csv", "ED/dev.csv", "ED/train_with_hard_negatives.csv"]
    if not all(os.path.exists(f) for f in required_files):
        print("[ERROR] Missing required dataset files. Please ensure train.csv, dev.csv, and train_with_hard_negatives.csv are in the ED/ directory.")
        return

    print(f"\n[INFO] Initializing {MODEL_NAME} on {DEVICE}...")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    dev_df     = pd.read_csv("ED/dev.csv")
    dev_loader = DataLoader(EvidenceDataset(dev_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = AdvancedDebertaCrossEncoder(MODEL_NAME, tokenizer).to(DEVICE)
    best_f1 = 0.0

    # STAGE 1: Train on augmented balanced data
    print("\n" + "="*50)
    print(" STAGE 1: Balanced Data Training")
    print("="*50)
    
    aug_s1_df    = augment_minority_class(pd.read_csv("ED/train.csv"), multiplier=2)
    s1_loader    = DataLoader(EvidenceDataset(aug_s1_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    cw_s1        = torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE)
    optimizer_s1 = build_optimizer(model, LEARNING_RATE)
    
    total_steps_s1 = (len(s1_loader) // ACCUMULATION_STEPS) * STAGE1_EPOCHS
    scheduler_s1   = get_linear_schedule_with_warmup(optimizer_s1, int(0.06 * total_steps_s1), total_steps_s1)

    epochs_no_improve = 0
    for epoch in range(STAGE1_EPOCHS):
        run_epoch(model, s1_loader, optimizer_s1, scheduler_s1, cw_s1, epoch+1, STAGE1_EPOCHS, "S1")
        m = evaluate(model, dev_loader)
        
        print(f"\n[S1 Validation] Epoch {epoch+1} | Macro F1: {m['macro_f1']:.4f} | Accuracy: {m['accuracy']:.4f}")
        print(m['report'])
        
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), "bestmodel.pt")
            print("[INFO] Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[INFO] Early stopping triggered for Stage 1 after {PATIENCE} epochs without improvement."); break

    # STAGE 2: Fine-tune on hard negatives
    print("\n" + "="*50)
    print(" STAGE 2: Hard Negatives Fine-Tuning")
    print("="*50)
    
    aug_s2_df = augment_minority_class(pd.read_csv("ED/train_with_hard_negatives.csv"), multiplier=5)
    s2_loader = DataLoader(EvidenceDataset(aug_s2_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    n0, n1 = (aug_s2_df["label"] == 0).sum(), (aug_s2_df["label"] == 1).sum()
    cw_s2  = torch.tensor([1.0, min(float(n0) / float(n1), 1.3)], dtype=torch.float32).to(DEVICE)
    
    optimizer_s2   = build_optimizer(model, LEARNING_RATE * STAGE2_LR_FACTOR)
    total_steps_s2 = (len(s2_loader) // ACCUMULATION_STEPS) * STAGE2_EPOCHS
    scheduler_s2   = get_linear_schedule_with_warmup(optimizer_s2, int(0.06 * total_steps_s2), total_steps_s2)

    epochs_no_improve = 0
    for epoch in range(STAGE2_EPOCHS):
        run_epoch(model, s2_loader, optimizer_s2, scheduler_s2, cw_s2, epoch+1, STAGE2_EPOCHS, "S2")
        m = evaluate(model, dev_loader)
        
        print(f"\n[S2 Validation] Epoch {epoch+1} | Macro F1: {m['macro_f1']:.4f} | Accuracy: {m['accuracy']:.4f}")
        print(m['report'])
        
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_deberta_evidence_model.pt")
            print("[INFO] Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[INFO] Early stopping triggered for Stage 2 after {PATIENCE} epochs without improvement."); break

    print(f" TRAINING COMPLETE | Best Macro F1: {best_f1:.4f}")
    print(f" Saved to -> best_deberta_evidence_model.pt")
   
if __name__ == "__main__":
    train()