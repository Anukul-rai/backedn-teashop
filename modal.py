# train_model_classification.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import pickle

# Load dataset
df = pd.read_csv("./data/teashop_sales_transactions.csv")

# Encode ProductID (convert text to numbers)
encoder = LabelEncoder()
df["ProductID"] = encoder.fit_transform(df["ProductID"])

# Create a SaleCategory target by grouping Amount into bins
# You can adjust bins based on your dataset range
df["SaleCategory"] = pd.cut(df["Amount"],
                            bins=[0, 100, 200, 300, 1000],  # define ranges
                            labels=[0, 1, 2, 3])           # assign category labels

# Features and target
X = df[["ProductID", "Amount"]]
y = df["SaleCategory"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print accuracy
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 2))

# Save the trained model as joblib
joblib.dump(model, "sales_model_classification.joblib")

# Save the trained model as pickle
with open("sales_model_classification.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoder to reuse later
joblib.dump(encoder, "label_encoder.joblib")
