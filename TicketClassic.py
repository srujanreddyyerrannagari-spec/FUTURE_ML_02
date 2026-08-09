# =========================================================
# Customer Support Ticket Classification & Priority Prediction
# =========================================================

# ---- imports ----
import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# ---- 1. Load the dataset ----
df = pd.read_csv("customer_support_tickets.csv")

# FIX: drop rows with missing ticket text so "nan" doesn't get treated as a word
df = df.dropna(subset=["Ticket Description", "Ticket Type", "Ticket Priority"])


# ---- 2. Text cleaning ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)          # remove punctuation
    text = re.sub(r"\d+", " ", text)              # remove standalone digits (optional)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)


df["clean_text"] = df["Ticket Description"].apply(clean_text)

# ---- 3. TF-IDF feature extraction ----
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_text"])


# =========================================================
# Reusable helper: train + evaluate any classifier for any target
# =========================================================
def train_and_evaluate(X, y, task_name, model):
    # FIX: stratify to keep class proportions consistent across train/test,
    # especially important since priority classes are imbalanced
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n===== {task_name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    # ---- Bonus: confusion matrix ----
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, preds, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{task_name} - Confusion Matrix")
    plt.tight_layout()
    safe_name = task_name.lower().replace(" ", "_")
    plt.savefig(f"{safe_name}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {safe_name}_confusion_matrix.png")

    return model, X_test, y_test, preds


# =========================================================
# 4. Category classification (compare a couple of algorithms)
# =========================================================
Y_category = df["Ticket Type"]

print("Comparing models for CATEGORY classification...")
category_candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
}

best_category_model = None
best_category_score = -1
best_category_name = None

for name, clf in category_candidates.items():
    trained_model, X_test_c, y_test_c, preds_c = train_and_evaluate(
        X, Y_category, f"Category - {name}", clf
    )
    score = accuracy_score(y_test_c, preds_c)
    if score > best_category_score:
        best_category_score = score
        best_category_model = trained_model
        best_category_name = name

print(f"\nBest category model: {best_category_name} (accuracy={best_category_score:.4f})")

# =========================================================
# 5. Priority classification (compare a couple of algorithms)
# =========================================================
Y_priority = df["Ticket Priority"]

print("\nComparing models for PRIORITY classification...")
priority_candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
}

best_priority_model = None
best_priority_score = -1
best_priority_name = None

for name, clf in priority_candidates.items():
    trained_model, X_test_p, y_test_p, preds_p = train_and_evaluate(
        X, Y_priority, f"Priority - {name}", clf
    )
    score = accuracy_score(y_test_p, preds_p)
    if score > best_priority_score:
        best_priority_score = score
        best_priority_model = trained_model
        best_priority_name = name

print(f"\nBest priority model: {best_priority_name} (accuracy={best_priority_score:.4f})")

# =========================================================
# 6. Save the best models + vectorizer (FIX: persistence)
# =========================================================
joblib.dump(best_category_model, "category_model.pkl")
joblib.dump(best_priority_model, "priority_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nSaved category_model.pkl, priority_model.pkl, vectorizer.pkl")

# =========================================================
# 7. Try it on a new ticket
# =========================================================
new_description = input("\nEnter a new support ticket description: ")
clean_description = clean_text(new_description)
vector_description = vectorizer.transform([clean_description])

predicted_category = best_category_model.predict(vector_description)[0]
predicted_priority = best_priority_model.predict(vector_description)[0]

print("\nPredicted Category:", predicted_category)
print("Predicted Priority:", predicted_priority)