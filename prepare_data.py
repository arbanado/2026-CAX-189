import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PRODUCTS = {
    "wine": "MntWines",
    "fruits": "MntFruits",
    "meat": "MntMeatProducts",
    "fish": "MntFishProducts",
    "sweets": "MntSweetProducts",
    "gold": "MntGoldProds",
}
CHANNELS = {
    "web": "NumWebPurchases",
    "catalog": "NumCatalogPurchases",
    "store": "NumStorePurchases",
}

INSTRUCTIONS = [
    "Analyze this customer record. Identify customer value, preferred product category, preferred shopping channel, campaign response, and one marketing recommendation.",
    "Review the customer market-research record and summarize purchasing behavior, strongest product preference, shopping-channel preference, response status, and a recommended marketing action.",
    "Using only the supplied customer data, produce a concise market-research profile with value level, top product, preferred channel, campaign response, and one actionable recommendation.",
]


def money(x):
    return f"${x:,.0f}"


def build_example(row, q1, q3, instruction):
    product_values = {name: int(row[col]) for name, col in PRODUCTS.items()}
    channel_values = {name: int(row[col]) for name, col in CHANNELS.items()}
    total_spend = sum(product_values.values())

    if total_spend <= q1:
        value_level = "low-value"
    elif total_spend >= q3:
        value_level = "high-value"
    else:
        value_level = "mid-value"

    top_product = max(product_values, key=product_values.get)
    top_channel = max(channel_values, key=channel_values.get)
    responded = int(row["Response"]) == 1

    if responded:
        recommendation = (
            f"Retain and grow this customer with personalized {top_product} offers through the {top_channel} channel, "
            "while avoiding unnecessary demographic assumptions."
        )
    else:
        recommendation = (
            f"Test a targeted {top_product} promotion through the {top_channel} channel and measure response before scaling, "
            "while avoiding unnecessary demographic assumptions."
        )

    context = (
        f"Education: {row['Education']}; Marital status: {row['Marital_Status']}; Income: {money(row['Income'])}; "
        f"Recency: {int(row['Recency'])} days; Wine spend: {money(row['MntWines'])}; Fruit spend: {money(row['MntFruits'])}; "
        f"Meat spend: {money(row['MntMeatProducts'])}; Fish spend: {money(row['MntFishProducts'])}; "
        f"Sweet spend: {money(row['MntSweetProducts'])}; Gold spend: {money(row['MntGoldProds'])}; "
        f"Web purchases: {int(row['NumWebPurchases'])}; Catalog purchases: {int(row['NumCatalogPurchases'])}; "
        f"Store purchases: {int(row['NumStorePurchases'])}; Latest campaign response: {int(row['Response'])}."
    )

    target = (
        f"Customer value: {value_level} (total product spending {money(total_spend)}). "
        f"Top product: {top_product} ({money(product_values[top_product])}). "
        f"Preferred channel: {top_channel} ({channel_values[top_channel]} purchases). "
        f"Campaign response: {'responded' if responded else 'did not respond'}. "
        f"Recommendation: {recommendation}"
    )
    return {"instruction": instruction, "context": context, "target": target}


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="marketing_campaign.csv")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, sep=None, engine="python")
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["Income"]).reset_index(drop=True)

    spend_cols = list(PRODUCTS.values())
    total_spend = df[spend_cols].sum(axis=1)
    q1, q3 = total_spend.quantile([0.25, 0.75]).tolist()

    records = [
        build_example(row, q1, q3, INSTRUCTIONS[i % len(INSTRUCTIONS)])
        for i, (_, row) in enumerate(df.iterrows())
    ]

    # 80% train, 10% validation, 10% held-out evaluation.
    train_records, temp_records = train_test_split(records, test_size=0.20, random_state=42)
    validation_records, evaluation_records = train_test_split(temp_records, test_size=0.50, random_state=42)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_records, out / "train.jsonl")
    save_jsonl(validation_records, out / "validation.jsonl")
    save_jsonl(evaluation_records, out / "evaluation.jsonl")

    print(f"Clean rows: {len(records)}")
    print(f"Training examples: {len(train_records)}")
    print(f"Validation examples: {len(validation_records)}")
    print(f"Held-out evaluation examples: {len(evaluation_records)}")
    print(f"Spending thresholds: Q1={q1:.2f}, Q3={q3:.2f}")


if __name__ == "__main__":
    main()
