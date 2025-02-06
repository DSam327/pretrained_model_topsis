import numpy as np
import pandas as pd

def topsis(df, weights, impact):
    data = df.iloc[:, 1:].values.astype(float)
    norm_data = data / np.linalg.norm(data, axis=0)
    weighted_data = norm_data * weights
    
    ideal_best = np.where(impact == 1, np.max(weighted_data, axis=0), np.min(weighted_data, axis=0))
    ideal_worst = np.where(impact == 1, np.min(weighted_data, axis=0), np.max(weighted_data, axis=0))
    
    dist_best = np.linalg.norm(weighted_data - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_data - ideal_worst, axis=1)
    
    score = dist_worst / (dist_best + dist_worst)
    df["TOPSIS Score"] = score
    df["Rank"] = df["TOPSIS Score"].rank(ascending=False)
    
    return df

data = {
    "Model": [
        "bertweet-base-sentiment-analysis", 
        "twitter-roberta-base-sentiment", 
        "distilbert-base-multilingual-cased-sentiments-student", 
        "twitter-xlm-roberta-base-sentiment"
    ],
    "Accuracy": [0.7063, 0.7170, 0.5209, 0.6952],
    "Precision": [0.7078, 0.7176, 0.5495, 0.7117],
    "Recall": [0.7260, 0.7354, 0.5732, 0.7252],
    "F1 Score": [0.7068, 0.7189, 0.4657, 0.6926]
}

df = pd.DataFrame(data)
weights = np.ones(df.shape[1] - 1) / (df.shape[1] - 1)
impact = np.ones(df.shape[1] - 1)  # Assuming all metrics are benefit criteria

df_topsis = topsis(df, weights, impact)
print(df_topsis)
