import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

numrows = 500 

namesfirst = ["Shivansh", "Aarav", "Vihaan", "Isha", "Ananya", "Kriti", "Rohan", "Kabir", "Meera", "Arjun"]
nameslast  = ["Sharma", "Gupta", "Singh", "Verma", "Patel", "Agarwal", "Mehta", "Reddy", "Iyer", "Das"]

rows = []

starttime = datetime(2025, 1, 1)

for i in range(numrows):
    name = f"{np.random.choice(namesfirst)} {np.random.choice(nameslast)}"


    income = np.random.randint(20000, 300001)

    expenses = int(income * np.random.uniform(0.3, 0.8))

    emis = int(income * np.random.uniform(0.0, 0.3))

    basescore = np.random.normal(720, 80)   
    creditscore = int(np.clip(basescore, 300, 900))

    loanrequested = int(income * np.random.uniform(3, 30))

    tenureyears = np.random.randint(1, 21)


    dti = (expenses + emis) / income

    emiratio = emis / income

    if creditscore >= 750:
        creditweight = 1.0
    elif creditscore >= 650:
        creditweight = 0.6
    else:
        creditweight = 0.2

    riskscore = (
        creditweight * 0.5 +
        (1 - dti) * 0.3 +
        (1 - emiratio) * 0.2
    ) * 100

    riskscore = max(0.0, min(100.0, riskscore))

    if riskscore >= 70:
        decision = "Approved"
    elif riskscore >= 50:
        decision = "Borderline"
    else:
        decision = "Rejected"

    timestamp = starttime + timedelta(minutes=i * 10)
    timestampstr = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": timestampstr,
        "name": name,
        "income": income,
        "expenses": expenses,
        "credit_score": creditscore,
        "emis": emis,
        "loan_requested": loanrequested,
        "tenure_years": tenureyears,
        "dti": dti,
        "emi_ratio": emiratio,
        "risk_score": riskscore,
        "decision": decision,
    }

    rows.append(row)

df = pd.DataFrame(rows)

csv_path = Path("loan_applications.csv")
df.to_csv(csv_path, index=False)

print(f"Generated {len(df)} synthetic rows and saved to {csv_path.resolve()}")
print(df.head())
