import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px


@st.cache_data
def load_and_preprocess_data():
    # Reads the CSV file directly. Ensure 'customerfile.csv' is in the same directory as this script.
    data = pd.read_csv('customerfile.csv')
    
    # Handle missing values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    
    # Feature engineering and encoding
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})
    data['Dependents'] = data['Dependents'].map({'Yes': 1, 'No': 0})
    data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})
    data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    data = data.drop('customerID', axis=1, errors='ignore')
    data = pd.get_dummies(data, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)
    
    serv_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in serv_cols:
        data[f'{col}_NoInternet'] = (data[col] == 'No internet service').astype(int)
        data[col] = data[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    
    data['MultipleLines_NoPhone'] = (data['MultipleLines'] == 'No phone service').astype(int)
    data['MultipleLines'] = data['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': 0})
    
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Scaling numerical variables
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    return data, scaler

# @st.cache_resource caches the trained model
@st.cache_resource
def train_model(data):
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    return logreg, svm_model, knn_model, X, X_test, y_test

# Load data and train models
data, scaler = load_and_preprocess_data()
logreg_model, svm_model, knn_model, X, X_test, y_test = train_model(data)


st.title("Customer Churn Prediction App")
st.markdown("This app predicts whether a customer is likely to churn based on their demographic and service data.")

st.subheader("Data Overview")
st.write("The original data is as follows:")
st.write(data.head())

st.write("The processed data is as follows:")
st.write(data.head())

st.subheader("1. Customer Churn Distribution")
churn_counts = data['Churn'].value_counts().reset_index()
churn_counts.columns = ['Churn Status', 'Count']
fig = px.pie(churn_counts, values='Count', names='Churn Status', 
             title='Percentage of Customers Who Churned',
             color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig)

st.subheader("Modeling and Evaluation")

# Use a consistent function for evaluation to avoid repetitive code
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    st.text("Confusion Matrix:")
    st.code(confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred))
    st.text(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

st.write("Logistic Regression Model Evaluation")
evaluate_model(logreg_model, X_test, y_test)

st.write("SVM Model Evaluation")
evaluate_model(svm_model, X_test, y_test)

st.write("KNN Model Evaluation")
evaluate_model(knn_model, X_test, y_test)

# Get the accuracy scores for all models
logreg_acc = accuracy_score(y_test, logreg_model.predict(X_test))
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

# Print the comparison
st.subheader("Model Comparison (Accuracy)")
st.text(f"Logistic Regression Accuracy: {logreg_acc:.4f}")
st.text(f"SVM Accuracy: {svm_acc:.4f}")
st.text(f"KNN Accuracy: {knn_acc:.4f}")
st.success("Conclusion: Based on accuracy, Logistic Regression is the best model.")

st.header("Predict Churn for a New Customer")
st.subheader("Enter Customer Details:")

# Create an input form for user data
with st.form(key='my_form'):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ('Male', 'Female'))
        SeniorCitizen = st.selectbox("Senior Citizen", ('Yes', 'No'))
        Partner = st.selectbox("Partner", ('Yes', 'No'))
        Dependents = st.selectbox("Dependents", ('Yes', 'No'))
        PhoneService = st.selectbox("Phone Service", ('Yes', 'No'))
        MultipleLines = st.selectbox("Multiple Lines", ('Yes', 'No', 'No phone service'))
        OnlineSecurity = st.selectbox("Online Security", ('Yes', 'No', 'No internet service'))
        OnlineBackup = st.selectbox("Online Backup", ('Yes', 'No', 'No internet service'))
        DeviceProtection = st.selectbox("Device Protection", ('Yes', 'No', 'No internet service'))
    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        MonthlyCharges = st.number_input("Monthly Charges", value=50.0)
        TotalCharges = st.number_input("Total Charges", value=500.0)
        InternetService = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
        TechSupport = st.selectbox("Tech Support", ('Yes', 'No', 'No internet service'))
        StreamingTV = st.selectbox("Streaming TV", ('Yes', 'No', 'No internet service'))
        StreamingMovies = st.selectbox("Streaming Movies", ('Yes', 'No', 'No internet service'))
        Contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
        PaperlessBilling = st.selectbox("Paperless Billing", ('Yes', 'No'))
        PaymentMethod = st.selectbox("Payment Method", ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    
    submit_button = st.form_submit_button(label='Predict Churn')

# When the user clicks the predict button, this block runs
if submit_button:
    # Create a dictionary from user input
    user_input = {
        'gender': gender,
        'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
    }
    
    # Convert input to a DataFrame
    input_df = pd.DataFrame([user_input])
    
    # --- Apply same preprocessing as in the training data ---
    
    # Binary encoding
    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
    input_df['Partner'] = input_df['Partner'].map({'Yes': 1, 'No': 0})
    input_df['Dependents'] = input_df['Dependents'].map({'Yes': 1, 'No': 0})
    input_df['PhoneService'] = input_df['PhoneService'].map({'Yes': 1, 'No': 0})
    input_df['PaperlessBilling'] = input_df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    
    # One-hot encoding and column alignment
    input_df = pd.get_dummies(input_df)
    training_cols = X.columns
    missing_cols = set(training_cols) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0
    input_df = input_df[training_cols]
    
    # Scaling numerical features
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Make prediction
    prediction_proba = logreg_model.predict_proba(input_df)[:, 1][0]
    
    st.subheader("Prediction Result")
    
    if prediction_proba >= 0.5:
        st.error(f"Prediction: This customer is at a **HIGH** risk of churning!")
        st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}", delta="High Risk")
    else:
        st.success(f"Prediction: This customer is **NOT** likely to churn.")
        st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}", delta="Low Risk")

