import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the car price dataset
# Replace 'car_data.csv' with your actual dataset filename
df = pd.read_csv('car_data.csv')

# Basic preprocessing (example: drop non-numeric columns, handle missing values)
df = df.select_dtypes(include=['number']).dropna()

# Features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
