import os, glob, json, re
import pandas as pd

DATASET_DIR = "Dataset"

def parse_json_file(path: str):
    raw = open(path, "r", encoding="utf-8").read().strip()

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    items = []
    ok = True
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            ok = False
            break
    if ok and items:
        return items


    fixed = re.sub(r",\s*([}\]])", r"\1", raw)
    if not fixed.startswith("["):
        fixed = "[" + fixed
    if not fixed.endswith("]"):
        fixed = fixed + "]"
    fixed = re.sub(r",\s*\]$", "]", fixed)  
    data = json.loads(fixed)
    return data if isinstance(data, list) else [data]

def build_dataframe():
    rows = []
    for journal_dir in sorted(os.listdir(DATASET_DIR)):
        full_dir = os.path.join(DATASET_DIR, journal_dir)
        if not os.path.isdir(full_dir):
            continue

        for f in glob.glob(os.path.join(full_dir, "*.json")):
            items = parse_json_file(f)
            for it in items:
                title = (it.get("title") or "").strip()
                abstract = (it.get("abstract") or "").strip()
                keywords = it.get("keywords") or []
                if isinstance(keywords, list):
                    kw = " ".join([str(k) for k in keywords])
                else:
                    kw = str(keywords)

                text = " ".join([title, abstract, kw]).strip()
                if len(text) < 20:
                    continue

                rows.append({
                    "text": text,
                    "label": journal_dir,                 
                    "journal_raw": it.get("journal"),     
                    "year": int(it.get("year")) if str(it.get("year","")).isdigit() else None,
                    "doi": it.get("doi"),
                    "source_file": os.path.basename(f),
                })


    df = pd.DataFrame(rows).dropna(subset=["label", "text"])
    return df

if __name__ == "__main__":
    df = build_dataframe()
    print(df.head())
    print("N =", len(df), "labels =", df["label"].nunique())
    print("Labels:", sorted(df["label"].unique()))
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/dataset.csv", index=False)
