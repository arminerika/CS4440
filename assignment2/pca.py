# -------------------------------------------------------------------------
# AUTHOR: Armin Erika Polanco
# FILENAME: pca.py
# SPECIFICATION: Drop-one-feature PCA; print PC1 variance per run and the max.
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
#df = ?
df = pd.read_csv("heart_disease_dataset.csv")
# keep only numeric predictors; drop a likely label column if present
df = df.select_dtypes(include=[np.number]).copy()
for y in ["HeartDisease", "heart_disease", "target", "label", "y"]:
    if y in df.columns:
        df.drop(columns=[y], inplace=True)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#Get the number of features
#--> add your Python code here
#num_features = ?
num_features = min(df.shape[1], 10)   # do at most 10 iterations
col_names = df.columns.tolist()
results = []

# Run PCA using 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here
    #reduced_data = ?
    reduced_data = np.delete(scaled_data, i, axis=1)

    # Run PCA on the reduced dataset
    # --> add your Python code here
    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    pc1_var = float(pca.explained_variance_ratio_[0])
    results.append((col_names[i], pc1_var))
    print(f"Dropped: {col_names[i]:>20s} | PC1 variance explained: {pc1_var:.4f}")

# Find the maximum PC1 variance
# --> add your Python code here
best_feature, best_variance = max(results, key=lambda x: x[1])

#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print("\nHighest PC1 variance found: {:.4f} when removing {}".format(best_variance, best_feature))