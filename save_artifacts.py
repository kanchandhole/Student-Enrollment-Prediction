"""
We generate encoder.pkl and label_encoder.pkl
"""

import re
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# Step 1 - Clean Purpose Function (same as notebook)

def clean_purpose(text):

    # Convert to string (handle nan)
    text = str(text)

    # Lowercase
    text = text.lower()

    # Replace s/w or s\w with software
    text = re.sub(r's[/\\]w', 'software', text)

    # Replace slash and backslash with ' and '
    text = re.sub(r'[/\\]', ' and ', text)

    # Replace + and & with ' and '
    text = re.sub(r'[+&]', ' and ', text)

    # Remove brackets and special characters except letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Fix common spelling mistakes
    corrections = {
        'intership': 'internship',
        'internnship': 'internship',
        'internishp': 'internship',
        'intenship': 'internship',
        'placment': 'placement',
        'regrding': 'regarding',
        'enqiry': 'enquiry',
        'enquri': 'enquiry',
        'enqurie': 'enquiry',
        'coustomer': 'customer',
        'salse': 'sales',
        'telesalse': 'telesales',
        'markrting': 'marketing',
        'testeing': 'testing',
        'cyberv': 'cyber',
        'concelling': 'counselling'
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text



# Step 2 - Load Dataset

df = pd.read_csv("1students_cleaned_checking_imblanced 1.csv")



# Step 3 - Same Cleaning as Notebook

df["FEEDBACK"] = df["FEEDBACK"].str.lower().str.strip()
df = df.drop(columns=["EMAIL", "Placement", "Training", "Attend_By", "FEEDBACK"], errors="ignore")
df["Purpose"] = df["Purpose"].fillna("unknown")
df["Registration"] = df["Registration"].fillna(df["Registration"].mode()[0])
df = df.drop(["Date", "Name", "Contact No"], axis=1, errors="ignore")
df = df.dropna()

# Apply clean_purpose function (same as notebook)
df["Purpose"] = df["Purpose"].apply(clean_purpose)

df["Registration"] = df["Registration"].str.lower().str.strip()

# Check unique purposes
print("Unique Purposes found:")
print(df["Purpose"].unique())



# Step 4 - Fit and Save OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoder.fit(df[["Purpose"]])
joblib.dump(encoder, "encoder.pkl")
print("\n encoder.pkl saved!")



# Step 5 - Fit and Save LabelEncoder
le = LabelEncoder()
le.fit(df["Registration"])
joblib.dump(le, "label_encoder.pkl")
print("label_encoder.pkl saved!")

print("\nClasses found:", list(le.classes_))
print("Purposes found:", list(encoder.categories_[0]))