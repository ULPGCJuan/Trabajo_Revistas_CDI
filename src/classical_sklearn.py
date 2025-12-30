import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("outputs/dataset.csv")
X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LinearSVC": LinearSVC(),
    "LogReg": LogisticRegression(max_iter=2000),
    "MultinomialNB": MultinomialNB(),
}

results = {}

for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), min_df=2)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, pred)
    rep = classification_report(y_test, pred, output_dict=True)
    results[name] = {"accuracy": acc, "report": rep}
    print("\n===", name, "===")
    print("Accuracy:", acc)
    print(classification_report(y_test, pred))

import os
os.makedirs("outputs", exist_ok=True)
with open("outputs/metrics_classical.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
