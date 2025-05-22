# notebooks/train_model.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# 1. Load data
df = pd.read_csv("data/Transaction_nuru_1.csv")

# 2. Drop non-numeric / non-model columns
X = df.drop(columns=["customer_id", "customer_name", "customer_account_category", "account_number"])

# 3. Fill any missing values (basic strategy)
X.fillna(0, inplace=True)

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train KMeans
k = 5  # Start with 5 clusters
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 6. Attach cluster labels back to original data
df["cluster"] = clusters

# 7. Save the model and scaler
joblib.dump(kmeans, "model/cluster_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

# 8. Save the clustered data (optional)
df.to_csv("data/Transaction_clustered.csv", index=False)

print("✅ Model trained and saved.")
