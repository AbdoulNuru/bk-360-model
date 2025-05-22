# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from app.customer_store import find_customer_by_account, find_customers_by_accounts, get_all_customers
from app.recommend_engine import score_customer


app = FastAPI()

origins = [
    "http://localhost:5173", "http://localhost:5174"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins (use with caution)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/customer/{account_number}")
def get_customer(account_number: str):
    results = find_customer_by_account(account_number)
    if results.empty:
        raise HTTPException(status_code=404, detail="Customer not found")

    row = results.iloc[0]
    cluster, recs = score_customer(row)

    return {
        "customer_id": str(row["customer_id"]),
        "customer_name": str(row["customer_name"]),
        "account_number": str(row["account_number"]),
        "cluster": cluster,
        "recommended_products": [
            {"name": name, "reason": reason} for name, reason in recs
        ]
    }

class BatchRequest(BaseModel):
    account_numbers: List[str]

@app.post("/recommend-batch")
def recommend_batch(data: BatchRequest):
    filtered = find_customers_by_accounts(data.account_numbers)
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No matching account numbers")

    output = []
    for _, row in filtered.iterrows():
        cluster, recs = score_customer(row)
        output.append({
            "customer_id": str(row["customer_id"]),
            "customer_name": str(row["customer_name"]),
            "account_number": str(row["account_number"]),
            "cluster": cluster,
            "recommended_products": [
                {"name": name, "reason": reason} for name, reason in recs
            ]
        })
    return output

@app.get("/recommend-all")
def recommend_all(page: int = 0, page_size: int = 100):
    start = page * page_size
    end = start + page_size

    all_data = get_all_customers().iloc[start:end]
    output = []

    for _, row in all_data.iterrows():
        cluster, recs = score_customer(row)
        output.append({
            "customer_id": str(row["customer_id"]),
            "customer_name": str(row["customer_name"]),
            "account_number": str(row["account_number"]),
            "cluster": int(cluster),
            "recommended_products": [
                {"name": str(name), "reason": str(reason)} for name, reason in recs
            ]
        })

    return {
        "page": page,
        "page_size": page_size,
        "records_returned": len(output),
        "data": output
    }

