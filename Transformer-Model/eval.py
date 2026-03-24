# ================================================================
# EVIDENCE DETECTION — DEMO INFERENCE (Google Colab Version)
# ================================================================
# HOW TO USE FOR PRESENTATION:
#
# 1. Open Google Colab (colab.research.google.com)
# 2. Runtime → Change runtime type → T4 GPU
# 3. Upload ONLY these two files using the Files panel (left sidebar):
#       - best_deberta_evidence_model.pt  (~1.5 GB)
#       - test.csv  (small)
# 4. Run all cells top to bottom
# 5. predictions.csv will appear in the Files panel — download it
#
# The model architecture is downloaded automatically from HuggingFace.
# No need to upload model config files.
# Total time: ~3 minutes on T4 GPU
# ================================================================


# ── Cell 1: Install dependencies ────────────────────────────────
# !pip install transformers sentencepiece -q


# ── Cell 2: Imports and precision fix ───────────────────────────
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# Must be set before any model loading
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
print(f"PyTorch : {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")


# ── Cell 3: Configuration ────────────────────────────────────────
# The model name — downloaded automatically from HuggingFace
# No local files needed
MODEL_NAME  = "microsoft/deberta-v3-large"
MODEL_PATH  = "best_deberta_evidence_model.pt"   # uploaded to Colab
INPUT_PATH  = "dev.csv"                         # uploaded to Colab
OUTPUT_PATH = "predictions.csv"
MAX_LEN     = 256
BATCH_SIZE  = 16   # T4 GPU can handle batch 16 easily


# ── Cell 4: Dataset ──────────────────────────────────────────────
class EvidenceDataset(Dataset):
    """
    Loads claim-evidence pairs for inference.
    Works with both dev.csv (has labels) and test.csv (no labels).
    Encoding: raw text, no prefixes — matches CSF training exactly.
    """
    def __init__(self, path, tokenizer, max_len):
        self.df = pd.read_csv(path)

        for col in ("Claim", "Evidence"):
            if col not in self.df.columns:
                raise ValueError(
                    f"Missing column '{col}'. Found: {self.df.columns.tolist()}"
                )

        before  = len(self.df)
        self.df = self.df.dropna(
            subset=["Claim", "Evidence"]
        ).reset_index(drop=True)
        if len(self.df) < before:
            print(f"  ⚠ Dropped {before - len(self.df)} NaN rows")

        self.tokenizer = tokenizer
        self.max_len   = max_len
        print(f"  Loaded {len(self.df)} samples from {path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        claim    = str(row["Claim"]).strip()
        evidence = str(row["Evidence"]).strip()

        enc = self.tokenizer(
            claim, evidence,
            add_special_tokens    = True,
            max_length            = self.max_len,
            padding               = "max_length",
            truncation            = True,
            return_attention_mask = True,
            return_tensors        = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }


# ── Cell 5: Model architecture ───────────────────────────────────
class AdvancedDebertaCrossEncoder(nn.Module):
    """
    CAFE: Hybrid cross-encoder
    DeBERTa-v3-large + 8-head Attention + 2-layer BiLSTM

    Feature vector: [CLS || BiLSTM_pooled || |u-v| || u*v] = 4 × 1024
    Classifier: Linear(4096, 2)
    """
    def __init__(self, model_name, tokenizer):
        super().__init__()
        # Downloaded automatically from HuggingFace on first run
        # Cached after that — subsequent runs are instant
        self.deberta = AutoModel.from_pretrained(model_name)
        H            = self.deberta.config.hidden_size
        self.sep_id  = tokenizer.sep_token_id

        self.attention  = nn.MultiheadAttention(
            embed_dim=H, num_heads=8, dropout=0.1, batch_first=True
        )
        self.attn_norm  = nn.LayerNorm(H)
        self.bilstm     = nn.LSTM(
            input_size=H, hidden_size=H // 2, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1,
        )
        self.dropouts   = nn.ModuleList(
            [nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)]
        )
        self.layer_norm = nn.LayerNorm(H * 4)
        self.classifier = nn.Linear(H * 4, 2)

    def forward(self, input_ids, attention_mask):
        out       = self.deberta(input_ids=input_ids,
                                 attention_mask=attention_mask)
        hidden    = out.last_hidden_state.to(torch.float32)
        cls_embed = hidden[:, 0, :]

        pad_mask    = (attention_mask == 0)
        attn_out, _ = self.attention(hidden, hidden, hidden,
                                     key_padding_mask=pad_mask)
        hidden      = self.attn_norm(hidden + attn_out)

        lstm_out, _ = self.bilstm(hidden)
        mask_exp    = attention_mask.unsqueeze(-1).float()
        lstm_pooled = (lstm_out * mask_exp).sum(1) \
                    / mask_exp.sum(1).clamp(min=1e-9)

        B = input_ids.size(0)
        diff_vecs, prod_vecs = [], []

        for i in range(B):
            sep_pos = (input_ids[i] == self.sep_id).nonzero(as_tuple=True)[0]
            if len(sep_pos) >= 2:
                s1, s2 = sep_pos[0].item(), sep_pos[1].item()
                c_tok  = hidden[i, 1:s1,    :]
                e_tok  = hidden[i, s1+1:s2, :]
            else:
                mid   = hidden.size(1) // 2
                c_tok = hidden[i, 1:mid, :]
                e_tok = hidden[i, mid:,  :]

            if c_tok.size(0) == 0: c_tok = hidden[i, [0], :]
            if e_tok.size(0) == 0: e_tok = hidden[i, [0], :]

            u = F.normalize(c_tok.mean(0), p=2, dim=-1)
            v = F.normalize(e_tok.mean(0), p=2, dim=-1)
            diff_vecs.append((u - v).abs())
            prod_vecs.append(u * v)

        feat   = self.layer_norm(
            torch.cat([cls_embed, lstm_pooled,
                       torch.stack(diff_vecs),
                       torch.stack(prod_vecs)], dim=1)
        )
        logits = sum(
            self.classifier(d(feat)) for d in self.dropouts
        ) / len(self.dropouts)

        return logits


# ── Cell 6: Load model and run inference ─────────────────────────
def run_demo():
    # Validate files
    if not os.path.exists(MODEL_PATH):
        print(f"✗ {MODEL_PATH} not found — upload it to Colab first")
        return
    if not os.path.exists(INPUT_PATH):
        print(f"✗ {INPUT_PATH} not found — upload it to Colab first")
        return

    print("="*60)
    print("  EVIDENCE DETECTION — INFERENCE")
    print("="*60)
    print(f"  Input  : {INPUT_PATH}")
    print(f"  Output : {OUTPUT_PATH}")
    print(f"  Device : {DEVICE}\n")

    # Tokeniser — downloaded automatically
    print("Step 1/4: Loading tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"✓ Tokeniser ready  (sep_id = {tokenizer.sep_token_id})")

    # Data
    print(f"\nStep 2/4: Loading {INPUT_PATH}...")
    dataset = EvidenceDataset(INPUT_PATH, tokenizer, MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)

    # Model — architecture downloaded, weights loaded from uploaded file
    print(f"\nStep 3/4: Building model and loading weights...")
    model      = AdvancedDebertaCrossEncoder(MODEL_NAME, tokenizer)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    # Architecture verification
    clf_shape = state_dict["classifier.weight"].shape
    clf_std   = state_dict["classifier.weight"].std().item()
    print(f"  Classifier : {clf_shape}  std={clf_std:.4f}")
    if clf_std < 0.001:
        print("  ⚠ WARNING: weights look untrained (std too low)")

    model.load_state_dict(state_dict, strict=True)
    model = model.to(torch.float32).to(DEVICE)
    model.eval()
    print(f"✓ Model ready on {DEVICE}")

    # Inference
    print(f"\nStep 4/4: Running inference on {len(dataset)} samples...")
    all_preds = []
    total     = len(loader)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)

            logits = model(ids, mask)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

            if (step + 1) % 20 == 0 or step == total - 1:
                pct = 100 * (step + 1) / total
                print(f"  Batch {step+1:4d}/{total}  [{pct:5.1f}%]")

    # Save
    pd.DataFrame({"label": all_preds}).to_csv(OUTPUT_PATH, index=False)

    n0 = all_preds.count(0)
    n1 = all_preds.count(1)
    print(f"\n{'='*60}")
    print(f"  ✓ Done!  Predictions saved → {OUTPUT_PATH}")
    print(f"  Class 0 (Not Evidence) : {n0}  ({100*n0/len(all_preds):.1f}%)")
    print(f"  Class 1 (Evidence)     : {n1}  ({100*n1/len(all_preds):.1f}%)")
    print(f"{'='*60}")

    # Download automatically in Colab
    try:
        from google.colab import files
        files.download(OUTPUT_PATH)
        print(f"✓ Download started")
    except ImportError:
        print(f"  (Not in Colab — find {OUTPUT_PATH} in current directory)")
    
# Run
run_demo()


#!pip install transformers==4.57.6 sentencepiece -q