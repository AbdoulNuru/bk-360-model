# app/recommend_engine.py

import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("model/cluster_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Features used in model
FEATURE_COLUMNS = [
    "total_txn_count",
    "avg_spend_amt",
    "total_spent",
    "has_paid_school",
    "has_paid_utility",
    "uses_mobile_money",
    "pays_taxes",
    "merchant_payments",
    "has_used_credit_card",
    "has_paid_tv_internet",
    "has_paid_gov_services",
    "sent_money_to_china",
    "has_paid_for_import_export"
]

# Full recommendation engine based on behavior + category
def get_recommendations(row):
    products = []
    category = str(row["customer_account_category"]).lower()

    # Soft-matching for primary agricultural producers
    if "agricul" in category:
        products.append(("Agri Loan", "Tailored for agricultural financing needs"))

    # Salary earners
    if any(keyword in category for keyword in ["salary earners public", "salary earners private", "salary ear priv"]):
        products.append(("BK Quick", "Suitable for salary advances up to RWF 500k"))
        products.append(("BK Quick Plus", "Higher limit loan with no collateral"))
        if row["avg_spend_amt"] > 50000:
            products.append(("Mortgage Loan", "Eligible based on income and expense level"))

    # Students
    if "student" in category:
        products.append(("Student Savings Account", "Ideal for managing low income and savings goals"))
        products.append(("Prepaid Card", "Smart and safe way to manage student expenses"))

    # BK Staff
    if "bk staff" in category:
        products.append(("BK Quick", "Special staff access to instant mobile loans"))
        products.append(("BK Quick Plus", "Larger limit with quicker approval"))
        products.append(("Mortgage Loan", "Staff-eligible housing finance solution"))

    # School fee payers
    if row["has_paid_school"]:
        products.append(("Tuza na BK", "Supports tuition fee payment with RWF 500k loan"))
        products.append(("Kira Kibondo", "Children’s saving account for long-term education goals"))

    # SME or merchant indicators
    if any(keyword in category for keyword in ["micro sme", "sole traders", "retail broker"]) or row["merchant_payments"]:
        products.append(("SME Stock Loan", "Support inventory or stock purchase"))
        products.append(("POS Device", "Enable seamless merchant payments"))

    # Credit card + high spender
    if row["has_used_credit_card"] and row["avg_spend_amt"] > 80000:
        products.append(("Secured Personal Loan", "Eligible due to card history and high spending"))
        products.append(("Credit Line", "Ongoing access to flexible credit"))

    # Import/export behavior
    if row["has_paid_for_import_export"]:
        products.append(("SME Bank Guarantee", "Secure trade operations and guarantee obligations"))

    # Utility + internet payers
    if row["has_paid_utility"] and row["has_paid_tv_internet"]:
        products.append(("Smart Save", "Digital savings based on active lifestyle"))
        products.append(("BK Wallet", "Ideal for digital transactions and mobile pay"))

    # Fallbacks based on general spending
    if row["has_paid_school"]:
        products.append(("Tuza na BK", "Supports tuition fee payment even without strong profile match"))

    if row["avg_spend_amt"] > 10000 and row["uses_mobile_money"]:
      products.append(("Bill Payments", "Customer uses mobile money frequently and can benefit from paying utilities through BK"))
      products.append(("Merchant Services", "Encourage use of BK POS and BK merchants for smoother digital payments"))


    # Final fallback if no matches
    if not products:
        products.append(("General Banking Package", "No clear pattern detected, but offer general BK services"))

    return products

# Score customer row (returns cluster + product recommendations)
def score_customer(row):
    input_row = row[FEATURE_COLUMNS].fillna(0)
    input_df = pd.DataFrame([input_row], columns=input_row.index)
    scaled = scaler.transform(input_df)
    cluster_id = int(model.predict(scaled)[0])
    recommendations = get_recommendations(row)
    return cluster_id, recommendations
