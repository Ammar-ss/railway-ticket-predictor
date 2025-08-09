import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#Load the dataset
df = pd.read_csv("Railway Ticket Confirmation.csv")

#Drop unused columns
df = df.drop(columns=["PNR Number", "Current Status", "Confirmation Status"])

#Handle dates
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'])
df['Booking Date'] = pd.to_datetime(df['Booking Date'])

# Add useful features from dates
df['Days Before Travel'] = (df['Date of Journey'] - df['Booking Date']).dt.days
df['Travel Month'] = df['Date of Journey'].dt.month
df['Day of Week'] = df['Date of Journey'].dt.dayofweek

#Define target and features
target = 'Number of Passengers'

# Drop target from features
features = df.drop(columns=[target])

#Identify column types
categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Ensure numeric date features are included
numeric_cols += ['Days Before Travel', 'Travel Month', 'Day of Week']
categorical_cols = list(set(categorical_cols) - set(['Date of Journey', 'Booking Date']))

#Build preprocessing and model pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

#Train-test split
X = features
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
model.fit(X_train, y_train)

#Evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))#Variance smaller absolute val = better
print("Root Mean Sq Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

#Plot
def plot_regression_results(y_test, y_pred_reg):
    plt.figure(figsize=(12,5))

    # Scatter plot: Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_reg, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal line
    plt.xlabel("Actual Number of Passengers")
    plt.ylabel("Predicted Number of Passengers")
    plt.title("Regression: Actual vs Predicted")

    # Residuals plot
    residuals = y_test - y_pred_reg
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_reg, residuals, alpha=0.5)
    plt.hlines(y=0, xmin=y_pred_reg.min(), xmax=y_pred_reg.max(), colors='r', linestyles='dashed')
    plt.xlabel("Predicted Number of Passengers")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Regression: Residuals Plot")

    plt.tight_layout()
    plt.show()
    
plot_regression_results(y_test, y_pred)
