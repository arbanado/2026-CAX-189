import argparse
import json
from pathlib import Path
import pandas as pd
from transformers import pipeline

BASE_MODEL = "google/flan-t5-small"
FINE_TUNED_MODEL = "./flan-market-research-final"


def load_examples(path, limit):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
            if len(examples) >= limit:
                break
    return examples


def generate(model_pipe, prompt):
    return model_pipe(
        prompt,
        max_new_tokens=160,
        do_sample=False,
    )[0]["generated_text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    base = pipeline("text2text-generation", model=BASE_MODEL, device=-1)
    tuned = pipeline("text2text-generation", model=FINE_TUNED_MODEL, device=-1)

    held_out = load_examples("data/evaluation.jsonl", args.limit)
    rows = []

    for i, ex in enumerate(held_out, start=1):
        prompt = f"Instruction: {ex['instruction']}\nContext: {ex['context']}"
        base_output = generate(base, prompt)
        tuned_output = generate(tuned, prompt)
        rows.append({
            "example": i,
            "prompt": prompt,
            "expected_response": ex["target"],
            "base_model_output": base_output,
            "fine_tuned_model_output": tuned_output,
        })
        print(f"\n=== HELD-OUT EXAMPLE {i} ===")
        print("EXPECTED:", ex["target"])
        print("BASE:", base_output)
        print("FINE-TUNED:", tuned_output)

    pd.DataFrame(rows).to_csv("model_comparison.csv", index=False)
    print("\nSaved comparison to model_comparison.csv")


if __name__ == "__main__":
    main()

