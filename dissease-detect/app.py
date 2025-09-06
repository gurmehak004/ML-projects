import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Data Preprocessing and Model Training ---
@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv(r'C:\Users\gurme\OneDrive\Documents\ml-intern\ml-projects\dissease-detect\diseasefile.csv')
    
    # Drop irrelevant columns
    # Your original code dropped date, country, and id. I'll include the necessary
    # logic here to make the mock data work, assuming those columns were present.
    irrelevant_cols = ['date', 'country', 'id']
    data = data.drop(columns=[col for col in irrelevant_cols if col in data.columns])

    # Convert age to years
    data['age'] = data['age'] / 365.25

    # One-hot encode nominal categorical features
    data = pd.get_dummies(data, columns=['occupation', 'gender'], drop_first=True)
    
    # Ensure binary columns are integers
    data = data.astype(int)

    # Scaling numerical values
    scaler = StandardScaler()
    num_cols = ['age', 'ap_hi', 'ap_lo', 'height', 'weight']
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    return data, scaler

@st.cache_resource
def train_model(data):
    X = data.drop('disease', axis=1)
    y = data['disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Logistic Regression model (best performing)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    return logreg, X.columns

# Load data and train model
data, scaler = load_and_preprocess_data()
logreg_model, training_cols = train_model(data)

# --- Streamlit UI for Prediction ---
st.title("Heart Disease Prediction App")
st.markdown("This app uses a Logistic Regression model to predict the presence of heart disease.")

st.header("Predict for a New Patient")
st.subheader("Enter Patient Details:")

# Create an input form for user data
with st.form(key='my_form'):
    col1, col2 = st.columns(2)
    with col1:
        age_days = st.slider("Age (days)", 10000, 30000, 20000)
        gender = st.selectbox("Gender", ('Male', 'Female'))
        ap_hi = st.slider("Systolic BP", 80, 200, 120)
        ap_lo = st.slider("Diastolic BP", 60, 120, 80)
        cholesterol = st.selectbox("Cholesterol", (1, 2, 3), format_func=lambda x: f"Level {x}")
        
    with col2:
        gluc = st.selectbox("Glucose Level", (1, 2, 3), format_func=lambda x: f"Level {x}")
        active = st.selectbox("Physically Active", ('Yes', 'No'))
        alco = st.selectbox("Alcohol Intake", ('Yes', 'No'))
        smoke = st.selectbox("Smokes", ('Yes', 'No'))
        height = st.slider("Height (cm)", 100, 200, 170)
        weight = st.slider("Weight (kg)", 30, 150, 70)
        occupation = st.selectbox("Occupation", ('Others', 'Engineer', 'Lawyer', 'Nurse', 'Teacher'))
    
    submit_button = st.form_submit_button(label='Predict Disease')

# When the user clicks the predict button, this block runs
if submit_button:
    # Create a dictionary from user input
    user_input = {
        'age': age_days / 365.25, # Convert to years
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'height': height,
        'weight': weight,
        'active': 1 if active == 'Yes' else 0,
        'alco': 1 if alco == 'Yes' else 0,
        'smoke': 1 if smoke == 'Yes' else 0,
        'gender_1': 1 if gender == 'Male' else 0, # Assuming '1' for male from original data
    }
    
    # Handle one-hot encoding for occupation
    for occ in ['Engineer', 'Lawyer', 'Nurse', 'Teacher', 'Others']:
        user_input[f'occupation_{occ}'] = 1 if occupation == occ else 0
    
    # Convert input to a DataFrame
    input_df = pd.DataFrame([user_input])
    
    # --- Apply same preprocessing as in the training data ---
    
    # One-hot encoding and column alignment
    # (Already handled manually above to align with the training data)
    
    # Scaling numerical features
    numerical_cols = ['age', 'ap_hi', 'ap_lo', 'height', 'weight']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Align columns with training data
    missing_cols = set(training_cols) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0
    input_df = input_df[training_cols]

    # Make prediction
    prediction_proba = logreg_model.predict_proba(input_df)[:, 1][0]
    
    st.subheader("Prediction Result")
    
    if prediction_proba >= 0.5:
        st.error(f"Prediction: This individual is at a **HIGH** risk of having heart disease!")
        st.metric(label="Disease Probability", value=f"{prediction_proba:.2%}", delta="High Risk")
    else:
        st.success(f"Prediction: This individual is **NOT** likely to have heart disease.")
        st.metric(label="Disease Probability", value=f"{prediction_proba:.2%}", delta="Low Risk")
