---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/username/project_name

---

# Model Card for Sameer-ED

<!-- Provide a quick summary of what the model is/does. -->

This model performs pairwise claim–evidence validity classification.
    Given a claim and a piece of evidence text, the model predicts whether the
    evidence supports or contradicts the claim.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

The model is based on a hybrid architecture combining
    DistilBERT embeddings with an ESIM-style bidirectional attention model.
    DistilBERT is used to generate contextual token representations for both the
    claim and the evidence, while BiLSTM and attention layers are used to model
    fine-grained interactions between the two texts. The model also includes a
    cross-encoded BERT signal based on the [CLS] token for global semantic
    information.

- **Developed by:** Sameer Mehmood
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** DistilBERT + ESIM (BiLSTM + attention)
- **Finetuned from model [optional]:** DistilBERT context embeddings used for ESIM model

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://github.com/coetaur0/ESIM
- **Paper or documentation:** https://aclanthology.org/P17-1152.pdf

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The model was trained on the ED (claim–evidence) classification
    dataset provided by COMP34812 training data. The dataset consists of
    +21K claim and evidence pairs labelled as valid or invalid (0/1).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate (distilBERT): 2e-5
      - train_batch_size: 32
      - eval_batch_size: 32
      - hidden_size: 128
      - compose_hidden_size: 256
      - max_length_claim: 100
      - max_length_evidence: 300
      - dropout: 0.3
      - num_epochs: 4
      - optimizer: Adam

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: ~2 hours on a single GPU
      - duration per training epoch: ~27 minutes
      - model size: ~260MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 6K pairs.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy
      - MCC

### Results


    The model obtained:
    - Accuracy =          0.8670
    - Macro Precision:    0.8331
    - Macro Recall:       0.8364
    - Macro F1:           0.8347
    - Weighted Precision: 0.8677
    - Weighted Recall:    0.8670
    - Weighted F1:        0.8673
    - Matthews Corrcoef:  0.6695
    

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB,
      - GPU: V100

### Software


      - Transformers 4.18.0
      - Pytorch 1.11.0+cu113
      - Python 3.9+
      - HuggingFace Transformers
      - NumPy
      - scikit-learn

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->


- Claim input limit = 100 tokens
- Evidence input limit = 300 tokens.

The model shows the following limitations:

- Struggles with implicit support or opposition: it often fails when evidence does not explicitly agree or disagree with the claim, relying instead on surface semantic similarity.
- Biased toward topic similarity: evidence mentioning the same topic as the claim may lead to false “valid” predictions.
- Confuses descriptive vs argumentative evidence: factual or descriptive statements are sometimes misclassified as supporting or refuting the claim.

These limitations highlight that the model captures semantic similarity well but may not fully reason over logical argument structures.


## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation with different values.
