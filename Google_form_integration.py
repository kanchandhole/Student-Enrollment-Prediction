"""
Percept Infosystem Consultancy - Google Form + Ml model Integration

this scripts"
1. reads new student responses from a Google Form (via Google Sheets API)
2. run Ml model prediction on "interseted Domain" column
3. saves prediction result back to google sheets.
4. run automatically every 30 secands chceking for new resposes.

google form questions:
1.  Email Address
2.  Full Name
3.  Contact Number
4.  City
5.  College Name
6.  Degree
7.  Branch
8.  Year Of Passout
9.  Interested Domain       <- ML model uses this column
10. Skills
11. Resume Upload
12. Why do you want this internship?
13. Are you available immediately?
"""

import re
import time
import joblib
import gspread
from google.oauth2.service_account import Credentials

#step 1: setup google sheets connection

#permissions needed to read and write google sheets
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",   # read and write sheets
    "https://www.googleapis.com/auth/drive"            # access google drive files
]

#load credentials from JSON file (downloaded from google cloud console)
#place credentials.josn in the same folder as this file
credentials = Credentials.from_service_account_file(
    "credentials.json",
    scopes=SCOPES
)

#connect to google sheets using credentials
client = gspread.authorize(credentials)

#this is the name of the google sheet when we linked our google form to google sheets, you can change it if your sheet has a different name
SHEET_NAME = "Percept Infosystem Consultancy - Internship Registration Form (Responses)"
sheet = client.open(SHEET_NAME).sheet1
 
print(f"Connected to Google Sheet: {SHEET_NAME}")

#step 2: load ML Model Artifacts
# Load the trained ML pipeline (StandardScaler + LogisticRegression)
pipeline = joblib.load("pipeline.pkl")
 
# Load the OneHotEncoder — converts "data science" → [1, 0, 0, ...]
encoder = joblib.load("encoder.pkl")
 
# Load the LabelEncoder — converts 0/1 → "not registered"/"registered"
le = joblib.load("label_encoder.pkl")
 
print("ML Model loaded successfully!")
# Step 3 - Clean Purpose Function (exactly same as EDA file)

def clean_purpose(text):
    """
    Cleans the student's Interested Domain input.
    Must be exactly the same as the cleaning done during model training.
    If cleaning is different → model gives wrong predictions.
 
    Example:
        "Data Science/AI"  →  "data science and ai"
        "intership python" →  "internship python"
        "s/w testing"      →  "software testing"
    """
 
    # Convert to string — handles NaN values
    text = str(text)
 
    # Convert to lowercase
    # "Data Science" → "data science"
    text = text.lower()
 
    # Replace s/w or s\w with software
    # "s/w testing" → "software testing"
    text = re.sub(r's[/\\]w', 'software', text)
 
    # Replace slash and backslash with ' and '
    # "python/java" → "python and java"
    text = re.sub(r'[/\\]', ' and ', text)
 
    # Replace + and & with ' and '
    # "python+django" → "python and django"
    text = re.sub(r'[+&]', ' and ', text)
 
    # Remove all special characters — keep only letters and spaces
    # "data@science!" → "data science"
    text = re.sub(r'[^a-z\s]', ' ', text)
 
    # Fix common spelling mistakes made by students in forms
    corrections = {
        'intership'   : 'internship',
        'internnship' : 'internship',
        'internishp'  : 'internship',
        'intenship'   : 'internship',
        'placment'    : 'placement',
        'regrding'    : 'regarding',
        'enqiry'      : 'enquiry',
        'enquri'      : 'enquiry',
        'enqurie'     : 'enquiry',
        'coustomer'   : 'customer',
        'salse'       : 'sales',
        'telesalse'   : 'telesales',
        'markrting'   : 'marketing',
        'testeing'    : 'testing',
        'cyberv'      : 'cyber',
        'concelling'  : 'counselling'
    }
 
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
 
    # Remove extra spaces
    # "data   science" → "data science"
    text = re.sub(r'\s+', ' ', text).strip()
 
    return text
 
 
# Step 4 - ML Prediction Function
def predict_registration(domain: str):
    """
    Takes student's Interested Domain as input.
    Cleans it, encodes it, runs ML model, returns prediction.
 
    Args:
        domain: Student's interested domain from Google Form
                e.g. "Data Science", "Python", "Web Development"
 
    Returns:
        Dictionary with prediction results
    """
 
    # Step 1: Clean the input — same as training data cleaning
    domain_clean = clean_purpose(domain)
 
    # Step 2: Encode text to numbers using saved OneHotEncoder
    # "data science" → [1, 0, 0, 0, ...]
    X_encoded = encoder.transform([[domain_clean]])
 
    # Step 3: Run ML model prediction
    pred_label = int(pipeline.predict(X_encoded)[0])          # 0 or 1
    proba      = pipeline.predict_proba(X_encoded)[0]         # [0.78, 0.22]
 
    # Step 4: Convert number back to text
    # 0 → "not registered", 1 → "registered"
    pred_str = le.inverse_transform([pred_label])[0]
 
    # Step 5: Calculate probability percentages
    prob_registered     = round(float(proba[1]) * 100, 2) if len(proba) > 1 else round((1 - float(proba[0])) * 100, 2)
    prob_not_registered = round(float(proba[0]) * 100, 2)
 
    return {
        "cleaned_domain"        : domain_clean,
        "prediction"            : pred_str,
        "probability_registered": f"{prob_registered}%",
        "probability_not_reg"   : f"{prob_not_registered}%"
    }
 
 

# Step 5 - Setup Extra Columns in Google Sheet

def setup_headers():
    """
    Google Form automatically creates these columns:
        Timestamp, Email Address, Full Name, Contact Number,
        City, College Name, Degree, Branch, Year Of Passout,
        Interested Domain, Skills, Resume Upload,
        Why do you want this internship?, Are you available immediately?
 
    This function adds these extra columns for ML predictions:
        Cleaned Domain          → cleaned version of Interested Domain
        ML Prediction           → registered / not registered
        Probability Registered  → e.g. 65.00%
        Probability Not Registered → e.g. 35.00%
        Processed               → YES when done — so we don't process again
    """
 
    # Get current headers from row 1
    headers = sheet.row_values(1)
    print(f"\nCurrent Sheet Headers: {headers}")
 
    # These are the extra columns we need to add
    extra_headers = [
        "Cleaned Domain",
        "ML Prediction",
        "Probability Registered",
        "Probability Not Registered",
        "Processed"
    ]
 
    # Add each missing header to the end of the sheet
    for header in extra_headers:
        if header not in headers:
            sheet.add_cols(1)
            next_col = len(sheet.row_values(1)) + 1
            sheet.update_cell(1, next_col, header)
            print(f"Added column: {header}")
 
    print("Sheet headers setup complete!\n")
 
 

# Step 6 - Process New Student Responses
def process_new_responses():
    """
    Reads all rows from Google Sheet.
    Finds rows that have NOT been processed yet (Processed != YES).
    Runs ML prediction on each unprocessed row.
    Saves results back to the sheet.
    Marks row as Processed = YES so it is not processed again.
    """
 
    # Get all data from the sheet
    all_rows = sheet.get_all_records()
    headers  = sheet.row_values(1)
    # Remove extra spaces from all headers
    headers = [h.strip() for h in headers]
    # Remove extra spaces from row keys
    all_rows = [{k.strip(): v for k, v in row.items()} for row in all_rows]
 
    # This is the exact column name from your Google Form
    # "Interested Domain" → student selects: Data Science / Web Development / Python / Java / Other
    domain_col_name = "Interested Domain"

 
    # Check if the column exists in the sheet
    if domain_col_name not in headers:
        print(f"Column '{domain_col_name}' not found in sheet!")
        print(f"Available columns are: {headers}")
        return
 
    # Get column index numbers
    # +1 because Google Sheets starts counting from 1 not 0
    domain_col_index       = headers.index(domain_col_name) + 1
    processed_col_index    = headers.index("Processed") + 1
    prediction_col_index   = headers.index("ML Prediction") + 1
    cleaned_col_index      = headers.index("Cleaned Domain") + 1
    prob_reg_col_index     = headers.index("Probability Registered") + 1
    prob_not_reg_col_index = headers.index("Probability Not Registered") + 1
 
    new_count = 0
 
    # Loop through each row — start from row 2 (row 1 is header)
    for i, row in enumerate(all_rows, start=2):
 
        # Check if this row was already processed
        # If YES → skip it — don't process again
        processed = row.get("Processed", "")
        if processed == "YES":
            continue
 
        # Get student's Interested Domain
        domain = row.get(domain_col_name, "")
        if not domain:
            continue  # skip empty rows
 
        # Get student info for logging
        student_name  = row.get("Full Name", "Unknown")
        student_email = row.get("Email Address", "Unknown")
        college       = row.get("College Name", "Unknown")
        degree        = row.get("Degree", "Unknown")
 
        print(f"\nProcessing Row {i}:")
        print(f"  Name    : {student_name}")
        print(f"  Email   : {student_email}")
        print(f"  College : {college}")
        print(f"  Degree  : {degree}")
        print(f"  Domain  : {domain}")
 
        # Run ML prediction on this student's domain
        result = predict_registration(domain)
 
        # Write prediction results back to Google Sheet
        sheet.update_cell(i, cleaned_col_index,       result["cleaned_domain"])
        sheet.update_cell(i, prediction_col_index,    result["prediction"])
        sheet.update_cell(i, prob_reg_col_index,      result["probability_registered"])
        sheet.update_cell(i, prob_not_reg_col_index,  result["probability_not_reg"])
        sheet.update_cell(i, processed_col_index,     "YES")   # mark as done
 
        print(f"Prediction: {result['prediction']} | Registered Probability: {result['probability_registered']}")
        new_count += 1
 
        # Wait 1 second between rows to avoid Google API rate limits
        # Google allows max 100 requests per 100 seconds
        time.sleep(1)
 
    # Summary
    if new_count == 0:
        print("No new responses found.")
    else:
        print(f"\n Successfully processed {new_count} new student responses!")
 
 
# Step 7 - Main Loop — Checks Every 30 Seconds

if __name__ == "__main__":
 
    print("\n Percept Infosystem - Google Form ML Integration Started!")
    print("Checking for new student responses every 30 seconds...")
    print("Press Ctrl+C to stop\n")
 
    # First setup the extra columns in sheet
    setup_headers()
 
    # Run forever — check every 30 seconds
    while True:
        try:
            print("\n Checking for new responses...")
            process_new_responses()
            print("\n Waiting 30 seconds before next check...")
            time.sleep(30)
 
        except KeyboardInterrupt:
            # User pressed Ctrl+C — stop gracefully
            print("\n Script stopped by user.")
            break
 
        except Exception as e:
            # Any other error — log it and retry after 30 seconds
            print(f"\n Error occurred: {e}")
            print("Will retry in 30 seconds...")
            time.sleep(30)