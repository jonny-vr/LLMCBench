from datasets import load_dataset
import pandas as pd
import os

# Change this to your working dir
BASE_DIR = "/mnt/lustre/work/geiger/gwb345/LLMCBench/data"

def convert_arc_to_csv(split, subset, out_path):
    dataset = load_dataset("ai2_arc", subset, split=split)
    rows = []

    for ex in dataset:
        question = ex["question"]
        choices = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        answer = ex["answerKey"]

        # Ensure fixed order: A, B, C, D
        choice_map = dict(zip(labels, choices))
        row = [question]
        for letter in ["A", "B", "C", "D"]:
            row.append(choice_map.get(letter, ""))
        row.append(answer)
        rows.append(row)

    columns = ["question", "A", "B", "C", "D", "answer"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out_path, index=False, header=False)
    print(f"Saved: {out_path}")


# Output paths
os.makedirs(f"{BASE_DIR}/ARC-E/dev", exist_ok=True)
os.makedirs(f"{BASE_DIR}/ARC-E/test", exist_ok=True)
os.makedirs(f"{BASE_DIR}/ARC-C/dev", exist_ok=True)
os.makedirs(f"{BASE_DIR}/ARC-C/test", exist_ok=True)

# ARC-Easy
convert_arc_to_csv("train", "ARC-Easy", f"{BASE_DIR}/ARC-E/dev/arc_easy_dev.csv")
convert_arc_to_csv("validation", "ARC-Easy", f"{BASE_DIR}/ARC-E/test/arc_easy_test.csv")

# ARC-Challenge
convert_arc_to_csv("train", "ARC-Challenge", f"{BASE_DIR}/ARC-C/dev/arc_challenge_dev.csv")
convert_arc_to_csv("validation", "ARC-Challenge", f"{BASE_DIR}/ARC-C/test/arc_challenge_test.csv")

