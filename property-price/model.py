import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# This file contains all the core data processing and model training logic.
# It is separated from the Streamlit UI for better code organization.

def load_data():
    """
    Loads the dataset, handles missing values, and performs one-hot encoding.
    Returns the original and encoded DataFrames.
    """
    try:
        data = pd.read_csv('propertyfile.csv')
    except FileNotFoundError:
        # A simple exit is used here since this function is not in the Streamlit app file.
        # The Streamlit app will handle the file not found error.
        raise FileNotFoundError("The 'propertyfile.csv' file was not found.")
    
    # Handle missing values by filling with the median
    data['total_bedrooms'].fillna(data['total_bedrooms'].median(), inplace=True)
    
    # One-hot encode the categorical feature
    data_encoded = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)
    return data, data_encoded

def train_models(data_encoded):
    """
    Trains both Simple and Multiple Linear Regression models.
    Returns the trained models and their predictions.
    """
    # Multiple Linear Regression
    X_multiple = data_encoded.drop('median_house_value', axis=1)
    y = data_encoded['median_house_value']
    
    X_train_multiple, X_test_multiple, y_train, y_test = train_test_split(
        X_multiple, y, test_size=0.2, random_state=42
    )
    
    model_multiple = LinearRegression()
    model_multiple.fit(X_train_multiple, y_train)
    predictions_multiple = model_multiple.predict(X_test_multiple)
    
    # Simple Linear Regression
    X_simple = data_encoded[['median_income']]
    X_train_simple, X_test_simple, _, _ = train_test_split(
        X_simple, y, test_size=0.2, random_state=42
    )
    
    model_simple = LinearRegression()
    model_simple.fit(X_train_simple, y_train)
    predictions_simple = model_simple.predict(X_test_simple)
    
    return (model_simple, model_multiple, y_test,
            predictions_simple, predictions_multiple)
