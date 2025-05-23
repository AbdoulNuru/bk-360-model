from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import threading
import time
import json
import os

from app.recommend_engine import get_recommendations

DATA_PATH = "data/Transaction_clustered.csv"
RECOMMEND_ALL_OUTPUT = "data/Recommendation_results.csv"

router = APIRouter()

cached_analytics = {}
last_updated = None
CACHE_DURATION = timedelta(minutes=5)

def generate_recommendations_file():
    try:
        df = pd.read_csv(DATA_PATH)
        df['recommendations'] = df.apply(get_recommendations, axis=1)
        df[['account_number', 'recommendations']].to_csv(RECOMMEND_ALL_OUTPUT, index=False)
        print("✅ Recommendation_results.csv generated.")
    except Exception as e:
        print(f"⚠️ Failed to generate Recommendation_results.csv: {e}")

def calculate_analytics():
    global cached_analytics, last_updated

    try:
        df = pd.read_csv(DATA_PATH)

        # Generate and merge recommendations file if not found
        if not os.path.exists(RECOMMEND_ALL_OUTPUT):
            generate_recommendations_file()

        if os.path.exists(RECOMMEND_ALL_OUTPUT):
            try:
                df_rec = pd.read_csv(RECOMMEND_ALL_OUTPUT)
                df_rec = df_rec.dropna(subset=['account_number'])
                df_rec['account_number'] = df_rec['account_number'].astype(str)
                df = df.merge(df_rec[['account_number', 'recommendations']], on='account_number', how='left')
            except Exception as merge_err:
                print(f"Error merging recommendation results: {merge_err}")

        df['recommendations'] = df['recommendations'].fillna("[]")

        df['recommendations_list'] = df['recommendations'].apply(lambda r: eval(r) if isinstance(r, str) else [])
        df['recommendation_count'] = df['recommendations_list'].apply(lambda r: len(r) if isinstance(r, list) else 0)

        total_customers = df['account_number'].nunique()
        total_recommendations = df['recommendation_count'].sum()
        avg_products_per_customer = round(total_recommendations / total_customers, 2) if total_customers else 0

        # Accurate cluster distribution using unique customers
        cluster_counts = df.groupby('cluster')['account_number'].nunique()
        cluster_distribution = [
            {
                "cluster": str(int(c)),
                "value": int(n),
                "percentage": f"{(n / total_customers) * 100:.2f}%"
            } for c, n in cluster_counts.sort_index().items()
        ]

        all_products = []
        for rlist in df['recommendations_list']:
            all_products.extend([r[0] for r in rlist if isinstance(r, (list, tuple)) and len(r) > 0])

        product_counter = Counter(all_products)
        product_recommendations = [
            {"name": name, "value": count, "description": "Top recommended product."}
            for name, count in product_counter.most_common(10)
        ]

        if 'score_segment' in df.columns:
            seg_counts = df['score_segment'].value_counts()
        else:
            seg_counts = pd.Series(dtype=int)
        customer_segments = [
            {"name": seg, "value": int(val)} for seg, val in seg_counts.items()
        ]

        cached_analytics = {
            "totalCustomers": int(total_customers),
            "totalRecommendations": int(total_recommendations),
            "conversionRate": None,
            "avgProductsPerCustomer": avg_products_per_customer,
            "clusterDistribution": cluster_distribution,
            "productRecommendations": product_recommendations,
            "customerSegments": customer_segments,
            "lastUpdated": datetime.utcnow().isoformat()
        }

        last_updated = datetime.utcnow()

    except Exception as e:
        print(f"Error calculating analytics: {e}")
        if not cached_analytics:
            raise HTTPException(status_code=500, detail="Unable to calculate analytics and no cache available.")

def start_analytics_updater():
    def run():
        while True:
            calculate_analytics()
            time.sleep(CACHE_DURATION.total_seconds())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

@router.get("/analytics")
def get_analytics():
    if not cached_analytics:
        calculate_analytics()
    return cached_analytics
