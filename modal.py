import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pickle
import os

# Define directory for model storage
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)  # Create folder if not exists

# Paths to save/load model and encoder
MODEL_FILE = os.path.join(MODEL_DIR, "sales_model_classification.joblib")
PICKLE_FILE = os.path.join(MODEL_DIR, "sales_model_classification.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.joblib")
DATA_FILE = "./data/teashop_sales_transactions.csv"

# Function to train model if it doesn't exist
def train_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
        print("Training model from scratch...")

        # Load dataset
        df = pd.read_csv(DATA_FILE)

        # Encode ProductID (text to numbers)
        encoder = LabelEncoder()
        df["ProductID"] = encoder.fit_transform(df["ProductID"])

        # Create SaleCategory target by grouping Amount into bins
        df["SaleCategory"] = pd.cut(
            df["Amount"],
            bins=[0, 100, 200, 300, 1000],
            labels=[0, 1, 2, 3]
        )

        # Features and target
        X = df[["ProductID", "Amount"]]
        y = df["SaleCategory"]

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Naive Bayes classifier
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Predict and print accuracy
        y_pred = model.predict(X_test)
        print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 2))

        # Save model and encoder to 'models' folder
        joblib.dump(model, MODEL_FILE)
        with open(PICKLE_FILE, "wb") as f:
            pickle.dump(model, f)
        joblib.dump(encoder, ENCODER_FILE)
        print("💾 Model and encoder saved in 'models' folder.")
    else:
        print("✅ Model and encoder already exist. Skipping training.")

# Train model if needed
train_model()

# Load trained model and encoder
model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)

# Prediction function for Python.NET or other use
def predict(product_id, amount):
    """
    Args:
        product_id (str): Product ID as string
        amount (float): Sale amount
    Returns:
        int: Predicted SaleCategory
    """
    # Encode product_id using the saved encoder
    product_id_encoded = encoder.transform([product_id])[0]
    features = np.array([[product_id_encoded, amount]])
    prediction = model.predict(features)
    return int(prediction[0])
