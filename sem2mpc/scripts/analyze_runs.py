# -*- coding: utf-8 -*-
import json, glob, csv, os

def load_metrics(pattern):
    rows = []
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        tag = os.path.basename(os.path.dirname(path))
        rows.append({
            "tag": tag,
            "N": m.get("N"),
            "end_position_error": m.get("end_position_error"),
            "min_obstacle_distance": m.get("min_obstacle_distance"),
            "solve_time_sec": m.get("solve_time_sec")
        })
    return rows

def main():
    rows = load_metrics("exp/**/_metrics.json")
    os.makedirs("reports", exist_ok=True)
    if not rows:
        print("No metrics found under exp/**/_metrics.json")
        return
    with open("reports/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("Saved reports/summary.csv with", len(rows), "rows")

if __name__ == "__main__":
    main()
