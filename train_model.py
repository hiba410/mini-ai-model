import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

data = {
    "text": [
        "Win money now", "Hello how are you",
        "Claim your prize", "Let's meet tomorrow",
        "Free offer just for you", "Good morning"
    ],
    "label": [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)

joblib.dump(model, "spam_model.pkl")

print("Model trained successfully!")