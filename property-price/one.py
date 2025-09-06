import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score


data=pd.read_csv(r'C:\Users\gurme\OneDrive\Documents\ml-intern\ml-projects\property-price\propertyfile.csv')
#to predict median house value in California districts based on features
# such as income, the number of rooms, geographical location, and proximity to the ocean.

#eda- summary stats 
print("First 5 rows of the dataset: ")
print(data.head())
print("\n Description of the dataset: ")
print(data.describe())
print("\n Information about the dataset: ")
print(data.info())
print("\n Missing values in each column count: ")
print(data.isnull().sum())
#handling null values
data['total_bedrooms'].fillna(data['total_bedrooms'].median(),inplace=True)
print("\n Missing values in each column count after handling nulls: ")
print(data.isnull().sum())

#visualizing in eda
print("Visualizations: ")
sns.set_style("whitegrid")
plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Key Numerical Features', fontsize=16)

plt.subplot(2, 3, 1)
sns.histplot(data['median_house_value'], kde=True)
plt.title('Median House Value')

plt.subplot(2, 3, 2)
sns.histplot(data['median_income'], kde=True)
plt.title('Median Income')

plt.subplot(2, 3, 3)
sns.histplot(data['total_rooms'], kde=True)
plt.title('Total Rooms')

plt.subplot(2, 3, 4)
sns.histplot(data['housing_median_age'], kde=True)
plt.title('Housing Median Age')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Scatter Plots for correlation
plt.figure(figsize=(10, 6))
plt.title('Median House Value vs. Median Income')
sns.scatterplot(x='median_income', y='median_house_value', data=data, alpha=0.5)
plt.show()

# Scatter plot of geographical location
plt.figure(figsize=(10, 8))
plt.title('Geographical Distribution of House Values')
sns.scatterplot(x='longitude', y='latitude', hue='median_house_value', data=data, palette='viridis', alpha=0.5)
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Box Plot for 'ocean_proximity'
plt.figure(figsize=(12, 6))
sns.boxplot(x='ocean_proximity', y='median_house_value', data=data)
plt.title('Median House Value by Ocean Proximity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
 #encoing the ocean proximity column-one hot encoding 
data_encoded=pd.get_dummies(data,columns=['ocean_proximity'],drop_first=True)
print("Original Data Columns:\n", data.columns)
print("\nNew Data Columns after One-Hot Encoding:\n", data_encoded.columns)
print("\nFirst 5 rows of the encoded data:\n", data_encoded.head())

#linear regression-simple , multiple
X_simple=data_encoded[['median_income']]
X_multiple=data_encoded.drop('median_house_value',axis=1)
y=data_encoded['median_house_value']

#splitting the data
X_simple_train,X_simple_test,y_train,y_test=train_test_split(X_simple,y,test_size=0.2,random_state=42)
X_multiple_train,X_mutiple_test,y_train,y_test=train_test_split(X_multiple,y,test_size=0.2,random_state=42)
#Simple Linear Regression 
linreg=LinearRegression()
linreg.fit(X_simple_train,y_train)
#prediction on test data 
pred=linreg.predict(X_simple_test)

#Multiple Linear Regression
linreg_multi=LinearRegression()
linreg_multi.fit(X_multiple_train,y_train)
pred_multi=linreg_multi.predict(X_mutiple_test)

#Evaluate 
mse_simple=mean_squared_error(y_test,pred)
r2_simple=r2_score(y_test,pred)
print(f"Simple Linear Regression - MSE: {mse_simple}, R2: {r2_simple}")

mse_multiple = mean_squared_error(y_test,pred_multi)
r2_multiple = r2_score(y_test,pred_multi)
print(f"Multiple Linear Regression - MSE: {mse_multiple}, R2: {r2_multiple}")