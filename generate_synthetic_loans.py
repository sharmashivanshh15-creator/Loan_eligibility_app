import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

num_rows = 500 

names_first = ["Shivansh", "Aarav", "Vihaan", "Isha", "Ananya", "Kriti", "Rohan", "Kabir", "Meera", "Arjun"]
names_last  = ["Sharma", "Gupta", "Singh", "Verma", "Patel", "Agarwal", "Mehta", "Reddy", "Iyer", "Das"]

rows = []

start_time = datetime(2025, 1, 1)

for i in range(num_rows):
    name = f"{np.random.choice(names_first)} {np.random.choice(names_last)}"


    income = np.random.randint(20000, 300001)

    expenses = int(income * np.random.uniform(0.3, 0.8))

    emis = int(income * np.random.uniform(0.0, 0.3))

    base_score = np.random.normal(720, 80)   
    credit_score = int(np.clip(base_score, 300, 900))

    loan_requested = int(income * np.random.uniform(3, 30))

    tenure_years = np.random.randint(1, 21)


    dti = (expenses + emis) / income

    emi_ratio = emis / income

    if credit_score >= 750:
        credit_weight = 1.0
    elif credit_score >= 650:
        credit_weight = 0.6
    else:
        credit_weight = 0.2

    risk_score = (
        credit_weight * 0.5 +
        (1 - dti) * 0.3 +
        (1 - emi_ratio) * 0.2
    ) * 100

    risk_score = max(0.0, min(100.0, risk_score))

    if risk_score >= 70:
        decision = "Approved"
    elif risk_score >= 50:
        decision = "Borderline"
    else:
        decision = "Rejected"

    timestamp = start_time + timedelta(minutes=i * 10)
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": timestamp_str,
        "name": name,
        "income": income,
        "expenses": expenses,
        "credit_score": credit_score,
        "emis": emis,
        "loan_requested": loan_requested,
        "tenure_years": tenure_years,
        "dti": dti,
        "emi_ratio": emi_ratio,
        "risk_score": risk_score,
        "decision": decision,
    }

    rows.append(row)

df = pd.DataFrame(rows)

csv_path = Path("loan_applications.csv")
df.to_csv(csv_path, index=False)

print(f"Generated {len(df)} synthetic rows and saved to {csv_path.resolve()}")
print(df.head())
