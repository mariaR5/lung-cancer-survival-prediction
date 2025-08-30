# Lung Cancer Survival Prediction Model

This project builds and deploys a Machine Learning model to predict the survival chances of lung cancer patients based on their clinical and demographic details. The model is trained in Google Colab and deployed as an interactive web app using Streamlit.

## Project Workflow

### Data Loading & Preprocessing

- Loaded dataset from Google Drive.
- Dropped unnecessary columns (id, diagnosis_date, end_treatment_date).
- Handled missing values (median for numeric, mode for categorical).
- Removed duplicates.
- Encoded categorical variables using LabelEncoder.
- Performed stratified sampling (15%) for manageable size.
- Applied SMOTE to handle class imbalance.

### Feature Scaling

- Standardized features using StandardScaler.

### Model Training

- Trained a Random Forest Classifier (n_estimators=24, random_state=42).

### Evaluation results:

- Accuracy: 74.06%

- Classification Report:
  
              precision    recall  f1-score   support

          0       0.73      0.77      0.75     20820
          1       0.75      0.72      0.73     20820


- Confusion Matrix:
  
    [[15951 4869]<br>[ 5931 14889]]

### Model Saving

- Saved trained model (model.pkl) and scaler (scaler.pkl) using joblib.
- Exported to Google Drive for persistence.

### App Deployment with Streamlit

- Built an interactive web app for predictions.
- Users can input patient details (age, gender, cancer stage, smoking status, BMI, etc.).
- Model predicts whether the patient is likely to survive or unlikely to survive, with a confidence score.
- Prediction history is stored during the session and can be downloaded as a CSV.

## How to Run
1. Clone Repository & Install Requirements
```
git clone <repo_url>
cd lung-cancer-survival
pip install -r requirements.txt
```
2. Run the Streamlit App

`streamlit run app.py`
