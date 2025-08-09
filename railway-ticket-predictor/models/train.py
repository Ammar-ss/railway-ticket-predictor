import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import joblib

from model import ConfirmationClassifier

def train_model():
    """Trains the classification model, evaluates it, and saves it."""
    # Load and preprocess data
    # Using the provided 'data' directory
    df = pd.read_csv('data/Railway Ticket Confirmation.csv')
    df = df.drop(columns=["PNR Number", "Current Status", "Waitlist Position"])

    # Feature Engineering
    df['Date of Journey'] = pd.to_datetime(df['Date of Journey'])
    df['Booking Date'] = pd.to_datetime(df['Booking Date'])
    df['Days Before Travel'] = (df['Date of Journey'] - df['Booking Date']).dt.days

    # Define features and target
    features = df.drop('Confirmation Status', axis=1)
    target = df['Confirmation Status'].apply(lambda x: 1 if x == 'Confirmed' else 0)

    # Identify column types
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    # Explicitly add datetime columns to categorical features for one-hot encoding
    datetime_cols = features.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    categorical_cols.extend(datetime_cols)
    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()


    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_processed.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_processed.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Initialize model, loss, and optimizer
    input_size = X_train_processed.shape[1]
    num_classes = 2
    model = ConfirmationClassifier(input_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("Starting model training...")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print("Training finished.")

    # Evaluation
    print("\nEvaluating model on the test set...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
        y_test_np = y_test.to_numpy()
        predicted_np = predicted.numpy()

        accuracy = accuracy_score(y_test_np, predicted_np)
        report = classification_report(y_test_np, predicted_np, target_names=['Not Confirmed', 'Confirmed'])
        
        print(f"\nAccuracy on Test Set: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)


    # Save the trained model and the preprocessor
    torch.save(model.state_dict(), 'models/ticket_classifier.pth')
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(input_size, 'models/input_size.joblib') # Save the input size
    print("\nModel, preprocessor, and input size saved successfully.")

if __name__ == '__main__':
    train_model()
