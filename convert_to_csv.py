import json, csv, os

IN  = "data/HellaSwag/jsonl/hellaswag_val.jsonl"
OUT = "data/HellaSwag/csv/hellaswag_val.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

with open(IN, "r", encoding="utf-8") as fi, open(OUT, "w", encoding="utf-8", newline="") as fo:
    writer = csv.writer(fo)
    for line in fi:
        ex = json.loads(line)
        writer.writerow([ex["ctx"]] + ex["endings"] + [ex["label"]])

print("Wrote:", OUT)

