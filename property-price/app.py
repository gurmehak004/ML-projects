import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
# Import the functions from your new model.py file
import model

# --- Page Configuration ---
st.set_page_config(
    page_title="Housing Price Prediction Dashboard",
    layout="wide",
)

# --- 1. Data Loading and Preprocessing ---
# We use st.cache_data and st.cache_resource on the function calls
# to ensure Streamlit's caching still works.
try:
    data, data_encoded = model.load_data()
except FileNotFoundError:
    st.error("The 'propertyfile.csv' file was not found. Please place it in the same directory.")
    st.stop()

# --- 2. Model Training ---
(model_simple, model_multiple, y_test,
 predictions_simple, predictions_multiple) = model.train_models(data_encoded)

# --- Streamlit UI ---
st.title("California Housing Price Prediction")
st.markdown("This dashboard showcases the exploratory data analysis (EDA) and model performance for predicting housing prices.")

# --- Tab Section ---
tab1, tab2, tab3 = st.tabs(["EDA & Data Overview", "Model Performance", "Make a Prediction"])

with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    
    st.subheader("1. Histograms of Key Features")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.histplot(data['median_house_value'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Median House Value')
    sns.histplot(data['median_income'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Median Income')
    sns.histplot(data['total_rooms'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Total Rooms')
    sns.histplot(data['housing_median_age'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Housing Median Age')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("2. Geographical Distribution of House Values")
    fig2 = plt.figure(figsize=(10, 8))
    sns.scatterplot(x='longitude', y='latitude', hue='median_house_value', data=data, palette='viridis', alpha=0.5)
    st.pyplot(fig2)
    
    st.subheader("3. Correlation Matrix")
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    correlation_matrix = data.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax3)
    st.pyplot(fig3)

    st.subheader("4. Box Plot of Median House Value by Ocean Proximity")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='ocean_proximity', y='median_house_value', data=data, ax=ax4)
    ax4.set_title('Median House Value by Ocean Proximity')
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

with tab2:
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Simple Linear Regression")
        mse_simple = mean_squared_error(y_test, predictions_simple)
        r2_simple = r2_score(y_test, predictions_simple)
        st.metric("MSE", f"{mse_simple:,.2f}")
        st.metric("R² Score", f"{r2_simple:.4f}")
        
    with col2:
        st.subheader("Multiple Linear Regression")
        mse_multiple = mean_squared_error(y_test, predictions_multiple)
        r2_multiple = r2_score(y_test, predictions_multiple)
        st.metric("MSE", f"{mse_multiple:,.2f}")
        st.metric("R² Score", f"{r2_multiple:.4f}")
        
    st.markdown("---")
    st.markdown("""
        ### Analysis of Results
        - The **Multiple Linear Regression model** has a significantly lower MSE and a higher R² score.
        - This indicates it's a **more accurate model** because it uses more features (including geographical data and ocean proximity), which explain a larger portion of the variance in housing prices.
    """)

with tab3:
    st.header("Make a Prediction")
    
    # Create sub-tabs for Simple vs. Multiple Regression
    tab_simple, tab_multiple = st.tabs(["Simple Linear Regression", "Multiple Linear Regression"])

    with tab_simple:
        st.subheader("Predict with Simple Linear Regression")
        st.markdown("This model uses only **Median Income** to predict the house value.")
        
        # Only one input field for this model
        simple_median_income = st.number_input("Median Income (in $10k)", value=3.81, format="%.2f", key="simple_income")
        
        if st.button("Predict with Simple Model", key="predict_simple"):
            # Create a DataFrame for the user input with a single feature
            input_data_simple = pd.DataFrame({'median_income': [simple_median_income]})
            
            # Make the prediction and display the result
            predicted_value = model_simple.predict(input_data_simple)
            st.markdown(f"### Predicted Median House Value: :green[${predicted_value[0]:,.2f}]")

    with tab_multiple:
        st.subheader("Predict with Multiple Linear Regression")
        
        st.markdown("""
        <style>
            .stMultiSelect div[data-baseweb="tag"] {
                background-color: lightblue !important;
                color: black !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("Select which features to include in the prediction.")
        
        # Get a list of all features for the multiselect
        features = list(data_encoded.drop('median_house_value', axis=1).columns)
        
        # Let the user select features
        selected_features = st.multiselect(
            "Select Features",
            options=features,
            default=[], # No features selected by default
            key="feature_multiselect"
        )
        
        # Create input widgets based on the selected features
        input_values = {}
        show_ocean_proximity_select = False
        for feature in selected_features:
            # Use the feature name as a unique key for each widget
            if feature == 'longitude':
                input_values[feature] = st.number_input(f"Longitude", value=-122.25, format="%.2f", key=f"input_{feature}")
            elif feature == 'latitude':
                input_values[feature] = st.number_input(f"Latitude", value=37.85, format="%.2f", key=f"input_{feature}")
            elif feature == 'median_income':
                input_values[feature] = st.number_input(f"Median Income (in $10k)", value=3.81, format="%.2f", key=f"input_{feature}")
            elif feature == 'housing_median_age':
                input_values[feature] = st.slider(f"Housing Median Age", 1, 52, 25, key=f"input_{feature}")
            elif feature == 'total_rooms':
                input_values[feature] = st.number_input(f"Total Rooms", value=1500, key=f"input_{feature}")
            elif feature == 'total_bedrooms':
                input_values[feature] = st.number_input(f"Total Bedrooms", value=250, key=f"input_{feature}")
            elif feature == 'population':
                input_values[feature] = st.number_input(f"Population", value=450, key=f"input_{feature}")
            elif feature == 'households':
                input_values[feature] = st.number_input(f"Households", value=190, key=f"input_{feature}")
            elif feature.startswith('ocean_proximity'):
                show_ocean_proximity_select = True
        
        # Display the selectbox outside the loop if any ocean_proximity feature is selected
        selected_ocean = None
        if show_ocean_proximity_select:
            ocean_proximity_options = ['<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY', 'INLAND', 'ISLAND']
            selected_ocean = st.selectbox("Ocean Proximity", options=ocean_proximity_options, key="ocean_proximity_select")
        
        if st.button("Predict with Multiple Model", key="predict_multiple"):
            # Create a full DataFrame with all features
            all_features_df = pd.DataFrame(columns=features)
            
            # Populate the DataFrame with user inputs for selected features
            row = {}
            for feature in features:
                if feature in selected_features and feature in input_values:
                    row[feature] = input_values[feature]
                elif feature.startswith('ocean_proximity'):
                    # Handle the one-hot encoded columns for the selected ocean proximity
                    if selected_ocean and feature.endswith(selected_ocean.replace(' ', '_').upper()):
                        row[feature] = 1
                    else:
                        row[feature] = 0
                else:
                    # Default values for unselected features
                    row[feature] = 0
            
            all_features_df = pd.DataFrame([row])
            
            # Make the prediction
            predicted_value = model_multiple.predict(all_features_df)
            
            st.markdown(f"### Predicted Median House Value: :green[${predicted_value[0]:,.2f}]")
