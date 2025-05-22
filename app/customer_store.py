# app/customer_store.py

import pandas as pd

# Load full clustered data
df = pd.read_csv("data/Transaction_clustered.csv")

def find_customer_by_account(account_number: str):
    result = df[df["account_number"].astype(str) == str(account_number)]
    return result

def find_customers_by_accounts(account_numbers):
    return df[df["account_number"].isin(account_numbers)]

def get_all_customers():
    return df.copy()
