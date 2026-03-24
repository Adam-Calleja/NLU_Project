import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Loading original training data...")
# Load your dataset
df = pd.read_csv('ED/train.csv')

claims = df['Claim'].tolist()
evidences = df['Evidence'].tolist()

print("Calculating TF-IDF similarities...")
vectorizer = TfidfVectorizer(stop_words='english')
# Fit on all evidence to create the vocabulary space
evidence_vectors = vectorizer.fit_transform(evidences)
claim_vectors = vectorizer.transform(claims)

# Calculate similarity between all claims and all evidence
similarity_matrix = cosine_similarity(claim_vectors, evidence_vectors)

hard_negatives = []

print("Mining hard negatives...")
for i, claim in enumerate(claims):
    # Get similarities for this claim against ALL evidence
    sim_scores = similarity_matrix[i]
    
    # Sort indices by highest similarity
    top_indices = np.argsort(sim_scores)[::-1]
    
    # Find the best matching evidence that is NOT the original paired evidence
    for idx in top_indices:
        if evidences[idx] != evidences[i]: # Ensure it's not the real answer
            hard_negatives.append({
                'Claim': claim,
                'Evidence': evidences[idx],
                'label': 0  # Label is 0 because it's a fake pair
            })
            break # Just take the top 1 hard negative per claim

# Create a new DataFrame and combine it with the original data
hard_negatives_df = pd.DataFrame(hard_negatives)
augmented_df = pd.concat([df, hard_negatives_df]).sample(frac=1).reset_index(drop=True)

augmented_df.to_csv('ED/train_with_hard_negatives.csv', index=False)
print(f"Generated {len(hard_negatives_df)} hard negatives. Saved to 'train_with_hard_negatives.csv'.")