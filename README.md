# Support Ticket Classification & Priority Prediction

An ML system that automatically reads incoming customer support tickets,
assigns them a **category** (e.g. Billing, Technical Issue, Account) and a
**priority level** (Low / Medium / High / Critical), so support teams spend
less time manually sorting tickets and more time solving them.

## Why this matters (business case)

Support teams typically lose significant time triaging tickets before any
actual problem-solving happens. A misrouted billing complaint sitting in the
technical queue, or a "my production system is down" ticket buried under
routine password resets, directly costs response time and customer trust.

This system automates the first, highest-leverage step: **read the ticket,
predict where it goes and how fast it needs attention** — before a human
ever looks at it.

## How categorization works

1. **Text cleaning** — the raw ticket text is lowercased, punctuation and
   stray digits are stripped, and common English filler words ("the", "is",
   "and"...) are removed. Words that signal negation ("not", "cannot") are
   deliberately kept, since they change meaning ("cannot log in" vs "can log in").
2. **Feature extraction (TF-IDF)** — cleaned text is converted into numeric
   vectors using TF-IDF with unigrams and bigrams, so the model can weigh
   both single words ("refund") and short phrases ("not working").
3. **Classification** — three algorithms (Logistic Regression, Multinomial
   Naive Bayes, Linear SVM) are trained on historical tickets and compared;
   the best-performing one is kept for production use.

## How priority is decided

Priority is predicted the same way — a separate classifier trained on the
same TF-IDF features, but targeting the `Ticket Priority` label instead of
category. In principle, priority should correlate with urgency signals in
the text (words like "urgent," "asap," "production down," "losing money")
combined with the category itself — outages and billing disputes trend
more urgent than general questions.

## Evaluation results

Both classifiers are evaluated using:
- **Accuracy** — overall percent correct
- **Precision / Recall / F1 per class** — because ticket categories and
  priorities are imbalanced, overall accuracy alone can be misleading
- **Confusion matrices** — to see exactly which categories/priorities get
  confused with each other

### Key insight from this run

On the current dataset, both the category and priority models perform
close to random-guessing levels, and the confusion matrices show no
diagonal dominance — predictions are scattered fairly evenly across all
classes rather than clustering on the correct label.

Investigating this, the ticket text and the category/priority labels in
this dataset **do not appear to carry a text-decodable relationship** —
i.e., the wording of a ticket doesn't reliably predict which category or
priority label it was tagged with. This is a common issue in synthetically
generated or auto-labeled ticket datasets found publicly (labels assigned
independently of content, for anonymization or synthetic-generation
reasons).

**Why this matters for the business case, not just the model:** it's a
reminder that automated ticket routing is only as good as the historical
labels it learns from. Before deploying a system like this in production,
a support team should audit a sample of historical tickets to confirm the
category/priority tags were actually assigned based on ticket content
(and not, e.g., which agent handled it, which queue it landed in by
default, or random assignment).

**What I'd recommend to a client in this situation:**
1. Pull a sample of real historical tickets with human-assigned labels
   and verify the labels actually reflect ticket content
2. Retrain on that verified sample — even a few hundred well-labeled
   tickets will outperform thousands of noisy ones
3. Once category accuracy is solid, priority prediction can be layered
   on top — it's the harder of the two tasks in any real deployment,
   since urgency is often implied rather than stated outright

## Tech stack

- Python, pandas, scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, Linear SVM)
- nltk (stopword removal)
- matplotlib / seaborn (confusion matrix visualization)
- joblib (model persistence)

## How to run

```bash
pip install pandas nltk scikit-learn matplotlib seaborn joblib
python TicketClassic.py
```

Trained models (`category_model.pkl`, `priority_model.pkl`,
`vectorizer.pkl`) are saved after training and can be reloaded without
retraining:

```python
import joblib
category_model = joblib.load("category_model.pkl")
priority_model = joblib.load("priority_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

new_ticket = ["My payment failed but I was still charged"]
vec = vectorizer.transform(new_ticket)
print(category_model.predict(vec), priority_model.predict(vec))
```
