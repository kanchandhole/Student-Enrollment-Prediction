🎓 Student Enrollment Prediction System

Percept Infosystem Consultancy — Internship Project

##  Project Overview:

During my internship at Percept Infosystem Consultancy, I worked on a student enrollment prediction project using real student inquiry data collected by the company. The dataset had limited features, with course interest (Purpose) being the most relevant column available for predicting whether a student would register or not.
The core challenge was that the team was manually tracking student inquiries and registrations, which was time-consuming and inefficient. There was no system in place to identify which students were likely to convert into actual registrations, making it difficult for counsellors to prioritize their follow-ups.
To solve this, I built an end-to-end machine learning system that automatically predicts whether a student will register based on their course interest. The model was trained on historical student data after thorough data cleaning, handling class imbalance using SMOTE, and encoding categorical features using OneHotEncoding.
The solution was further integrated with the company's Google Form, so that whenever a new student submits an inquiry, the ML model automatically predicts their registration likelihood and saves the result back to Google Sheets — eliminating the need for any manual tracking.

##  Problem Statement:

1. A training institute was collecting student inquiry data manually — recording student names, contact details, their reason for visiting (Purpose), and whether they registered or not. Over time, this data became messy and unreliable. The Purpose field was filled in freehand by different staff members, so the same intent was written in dozens of different ways — spelling mistakes like intership, placment, markrting, abbreviations like s/w for software, mixed use of /, +, & symbols, and inconsistent casing. There were also duplicate student entries, missing contact numbers, and date formats that were all over the place. To fix this, a complete data cleaning pipeline was built — standardizing text, correcting spelling mistakes using a custom clean_purpose() function, removing duplicates by prioritizing records with email addresses, fixing date formats, and dropping rows with missing critical information.

2. When we tried to analyze this data, the FEEDBACK column was causing data leakage — it directly reflected whether a student registered or not, which would have made the model cheat during training. So it had to be carefully identified and removed before modeling to ensure the ML model learned only from genuine inquiry patterns.

3. Even during EDA, simple things like understanding which counselor handled the most students was not straightforward — the same counselor Pooja mam was recorded as pooja ma'am, pooja m, Pooja mam  (with trailing space), and several other variations. All counselor name variations were standardized so the analysis could actually reflect real patterns.

4. The root cause of all this inconsistency was manual data entry — staff were adding student details directly into an Excel sheet by hand, which led to all the spelling mistakes, format mismatches, and duplicate records in the first place. To solve this at the source, a Google Form integration was built so that from now on, student inquiries are collected through a structured form, automatically predicted by the ML model, and written back to the sheet — eliminating manual entry and ensuring clean, consistent data going forward.

##  Solution

Built an end-to-end ML system that cleans and learns from historical student data, predicts whether a new student inquiry will convert into an enrollment, and automatically runs on every new Google Form submission in real time — writing the prediction result directly back into the Google Sheet every 30 seconds, without any manual effort.

##  What I Built:

* Cleaned and analyzed real student enrollment data from Percept Infosystem
* Built a Machine Learning model(Logistic Regression, Random Forest, Decision Tree) to predict student registration
* Created a REST API using FastAPI so the model can be used by any application
* Built a web UI using Streamlit for easy use by non-technical staff
* Integrated with the company's Google Form — predictions happen automatically when a student submits the form


##  Project Structure

```
EDA Student Enrollments Prediction
│
├── main.py                                        → FastAPI backend
├── app.py                                         → Streamlit frontend
├── save_artifacts.py                              → Saves encoder and label encoder
├── google_form_integration.py                     → Auto-predicts from Google Form
├── pipeline.pkl                                   → Trained ML model
├── encoder.pkl                                    → OneHotEncoder for Purpose
├── label_encoder.pkl                              → LabelEncoder for Registration
├── requirements.txt                               → Python dependencies
├── EDA_student_enrollments_prediction_.ipynb      → EDA and ML model notebook
├── Data_Cleaning_student_enrollments_prediction.ipynb → Data cleaning notebook
├── 1students_cleaned_checking_imblanced 1.csv     → Cleaned dataset
├── student data modified 1update.csv              → Raw original dataset
├── percept logo.png                               → Institute logo
├── .gitignore                                     → Files ignored by GitHub
└── README.md                                      → Project documentation
```

##   How It Works
```
 📁 Data Cleaning — Data_Cleaning_student_enrollments_prediction.ipynb

Raw student data loaded (student data modified 1update.csv)
        ↓
Dropped irrelevant columns (Sr NO, Sadar Office)
        ↓
Removed rows with missing contact numbers
        ↓
Standardized text columns (lowercase, strip spaces)
        ↓
Fixed inconsistent date formats & forward-filled missing dates
        ↓
Removed duplicate student entries (kept rows with email)
        ↓
Built clean_purpose() function → fixed spelling mistakes, symbols, abbreviations
        ↓
Used FEEDBACK column keywords to label students as Not Registered
        ↓
Dropped FEEDBACK column to prevent data leakage
        ↓
Saved clean output → 1students_cleaned_checking_imblanced 1.csv 


📊 EDA — EDA_student_enrollments_prediction_.ipynb

Loaded cleaned dataset (1students_cleaned_checking_imblanced 1.csv)
        ↓
Checked dataset shape, size, head, tail, describe
        ↓
Checked null values and missing value percentage
        ↓
Standardized FEEDBACK column (lowercase, strip spaces)
        ↓
Checked FEEDBACK percentage distribution
        ↓
── EDA ──────────────────────────────────────────
        ↓
Filled missing Attend_By values with "Unknown"
        ↓
Standardized all counselor name variations
(pooja ma'am, pooja m → Pooja mam)
(Walk-in, Walkin, walk in → Direct walk in)
        ↓
Plotted counselor-wise student distribution chart
        ↓
Dropped unnecessary columns
(EMAIL, Placement, Training, Attend_By)
        ↓
Dropped FEEDBACK column → prevent data leakage
        ↓
Converted Date column to datetime format
        ↓
Analyzed student inquiries by month & year
        ↓
Plotted bar chart → Student Inquiries per Year
        ↓
Plotted line chart → Trend of Inquiries by Year
        ↓
Conclusion: Most inquiries in 2024, least in 2026
        ↓
Analyzed top 10 most common Purpose values
        ↓
Plotted horizontal bar chart → Top 10 Inquiry Purposes
        ↓
Plotted Registration distribution chart
        ↓
Conclusion: Direct walk-in generates highest leads
        ↓
Checked unique Purpose values & grouped by Purpose size
        ↓
── PRE-PROCESSING ───────────────────────────────
        ↓
Filled missing Purpose values with "Unknown"
        ↓
Checked Registration value counts & distribution %
        ↓
Filled missing Registration values with mode value
        ↓
Saved intermediate file → 1students_drop_column.csv
        ↓
Dropped Date, Name, Contact No columns
        ↓
Dropped remaining null rows
        ↓
Cleaned Purpose & Registration columns
(lowercase, strip spaces)
        ↓
── FEATURE SELECTION ────────────────────────────
        ↓
Defined X = Purpose (only meaningful feature)
Defined y = Registration (target variable)
        ↓
── ENCODING ─────────────────────────────────────
        ↓
Applied OneHotEncoder on Purpose column (X)
→ sparse_output=False (dense array)
        ↓
Applied LabelEncoder on Registration column (y)
        ↓
── TRAIN-TEST SPLIT ─────────────────────────────
        ↓
Split data → 80% Train, 20% Test (random_state=42)
        ↓
── HANDLE CLASS IMBALANCE ───────────────────────
        ↓
Checked class distribution before SMOTE
        ↓
Applied SMOTE on training data only
(to avoid data leakage on test data)
        ↓
Checked class distribution after SMOTE
        ↓
── FEATURE SCALING ──────────────────────────────
        ↓
Applied StandardScaler on training data (fit + transform)
Applied StandardScaler on test data (transform only)
        ↓
── MODEL TRAINING ───────────────────────────────
        ↓
Trained Logistic Regression
        ↓
Trained Decision Tree Classifier
        ↓
Trained Random Forest Classifier (100 estimators)
        ↓
── EVALUATION (Round 1) ─────────────────────────
        ↓
Evaluated all 3 models →
Precision, Recall, F1 Score, Accuracy, Classification Report
        ↓
Results:
Logistic Regression → Accuracy: 91.7%, Class 1 F1: 0.58
Decision Tree       → Accuracy: 92.6%, Class 1 F1: 0.61
Random Forest       → Accuracy: 92.6%, Class 1 F1: 0.61
        ↓
Conclusion: All models struggle with minority class
(Registered students recall only 54%)
        ↓
── IMPROVEMENT ATTEMPTS ─────────────────────────
        ↓
Tried Logistic Regression with class_weight='balanced'
        ↓
Tried Random Forest with class_weight='balanced'
        ↓
Applied RandomOverSampler on training data
        ↓
Retrained Logistic Regression on oversampled data
        ↓
Retrained Random Forest on oversampled data
        ↓
Final Conclusion: No significant improvement
→ Limited features cause performance ceiling
        ↓
── CONFUSION MATRIX ─────────────────────────────
        ↓
Plotted Confusion Matrix → Random Forest
        ↓
Plotted Confusion Matrix → Logistic Regression
        ↓
Plotted Confusion Matrix → Decision Tree
        ↓
── HYPERPARAMETER TUNING ────────────────────────
        ↓
Applied GridSearchCV on Random Forest
(n_estimators, max_depth, min_samples_split)
5-fold cross validation
        ↓
Found Best Parameters
        ↓
── FINAL MODEL PIPELINE ─────────────────────────
        ↓
Built sklearn Pipeline →
StandardScaler + Logistic Regression
        ↓
Trained pipeline on X_train (original split)
        ↓
Saved → pipeline.pkl 
        ↓
Retrained pipeline on oversampled data (X_train_over)
        ↓
Saved updated → pipeline.pkl 


🤖 ML Model — save_artifacts.py

Loaded cleaned dataset (1students_cleaned_checking_imblanced 1.csv)
        ↓
Dropped unnecessary columns (EMAIL, Placement, Training, Attend_By, FEEDBACK)
        ↓
Filled missing Purpose values with "unknown"
        ↓
Filled missing Registration values with most frequent value (mode)
        ↓
Dropped Date, Name, Contact No columns
        ↓
Applied clean_purpose() on Purpose column
        ↓
Standardized Registration column (lowercase, strip spaces)
        ↓
Fitted OneHotEncoder on Purpose column → saved as encoder.pkl 
        ↓
Fitted LabelEncoder on Registration column → saved as label_encoder.pkl
```

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas | Data cleaning and analysis |
| Scikit-learn | ML model training |
| FastAPI | REST API development |
| Streamlit | Web UI for predictions |
| gspread | Google Sheets integration |
| Joblib | Save and load ML model |
| SMOTE | Handle class imbalance in data |


## Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 91.7% |
| Decision Tree | 92.6% |
| Random Forest | 92.6% |

Best Model:  Decision Tree / Random Forest with 92.6% accuracy

##   Setup Instructions:

## Step 1 — Clone Repository
git clone https://github.com/kanchandhole/Student-Enrollment-Prediction.git
cd student-enrollment-prediction

## Step 2 — Install Dependencies

pip install -r requirements.txt

## Step 3 — Run Save Artifacts (Only Once)

python save_artifacts.py

## Step 4 — Setup Google Cloud (For Google Form Integration)

Follow every step carefully to get your own credentials.json file.

## PART A — Create Google Cloud Project

1. Go to: https://console.cloud.google.com
2. Sign in with your Google account
3. Click "Select a Project" at the top
4. Click "New Project"
5. Project name: student-enrollment
6. Click "Create"
7. Wait for project to be created
8. Make sure your new project is selected at the top

## PART B — Enable Google Sheets API

1. In left menu click "APIs & Services"
2. Click "Library"
3. In search box type: Google Sheets API
4. Click on "Google Sheets API"
5. Click "Enable" button
6. Wait for it to enable

## PART C — Enable Google Drive API

1. Go back to "Library"
2. In search box type: Google Drive API
3. Click on "Google Drive API"
4. Click "Enable" button
5. Wait for it to enable

## PART D — Create Service Account

1. In left menu click "APIs & Services"
2. Click "Credentials"
3. Click "+ Create Credentials" at top
4. Select "Service Account"
5. Fill in the form:
   Service account name: student-enrollment-service
   Service account ID: student-enrollment-service (auto filled)
   Description: ML model access to Google Sheets
6. Click "Create and Continue"
7. On next screen (Grant Access) → Click "Continue" (skip this)
8. On next screen (Grant Users) → Click "Done" (skip this)

## PART E — Download credentials.json

1. You will see your service account listed:
   student-enrollment-service@student-enrollment-xxxxx.iam.gserviceaccount.com
2. Click on that email link
3. Click "Keys" tab at the top
4. Click "Add Key"
5. Click "Create New Key"
6. Select "JSON"
7. Click "Create"
8. A JSON file will download automatically
9. Rename that file to: credentials.json
10. Place credentials.json in your project folder:
    C:\your-project-folder\credentials.json


## PART F — Copy Service Account Email

1. Open credentials.json file in any text editor
2. Find the "client_email" field:

   "client_email": "student-enrollment-service@student-enrollment-xxxxx.iam.gserviceaccount.com"

3. Copy that email address — you will need it in next step


## PART G — Link Google Form to Google Sheets

1. Open your Google Form
2. Click "Responses" tab at the top
3. Click the green Sheets icon 🟢
4. Select "Create a new spreadsheet"
5. Give it a name (remember this name exactly)
6. Click "Create"
7. Google Sheet will open automatically with all form columns

## PART H — Share Google Sheet with Service Account

1. Open your Google Sheet
2. Click "Share" button (top right corner)
3. In the "Add people" box paste the email you copied in PART F:
   student-enrollment-service@student-enrollment-xxxxx.iam.gserviceaccount.com
4. Make sure role is set to "Editor"
5. Uncheck "Notify people"
6. Click "Share"

## PART I — Update Sheet Name in Code

1. Open google_form_integration.py
2. Find this line:
   SHEET_NAME = "Student Enrollment Responses"
3. Replace with your exact Google Sheet name from PART G:
   SHEET_NAME = "Your Exact Sheet Name Here"
4. Save the file

##  How to Run

Open 3 terminals and run one command in each:

## Terminal 1 — FastAPI Backend:

uvicorn main:app --reload

## Terminal 2 — Streamlit Frontend:

streamlit run app.py

## Terminal 3 — Google Form Integration:

python google_form_integration.py

##  Access Links

```
What                          URL
Streamlit UI                 http://localhost:8501
FastAPI Swagger              http://127.0.0.1:8000/docs
Health Check                 http://127.0.0.1:8000/health
```

##  How Google Form Integration Works
```
Student fills Google Form
        ↓
Response automatically saved in Google Sheets
        ↓
Python script checks for new responses every 30 seconds
        ↓
ML Model predicts registration likelihood
        ↓
Prediction saved back to Google Sheet 
No manual work needed!
```

## Important Notes

* Never share your credentials.json — it contains your Google Cloud private key
* Every user must create their own credentials.json following PART A to PART I above
* Run save_artifacts.py once before starting the API
* All 3 terminals must stay open while using the project


## About

Intern: Kanchan Charandas Dhole
Company: Percept Infosystem Consultancy, Nagpur
Duration: 6 month
Project Type: End-to-End Machine Learning Project