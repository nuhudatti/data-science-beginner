import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
]
df = pd.read_csv("dataset/house_prices.csv", delim_whitespace=True, names=column_names)

# Step 2: Basic Exploration
print("\n✅ First 5 Rows of the Dataset:")
print(df.head())

print("\n✅ Dataset Info:")
print(df.info())

print("\n✅ Missing Values:")
print(df.isnull().sum())

print("\n✅ Basic Statistics:")
print(df.describe())

# Step 3: Clean the dataset
df.dropna(inplace=True)

# Remove outliers in price
Q1 = df['MEDV'].quantile(0.25)
Q3 = df['MEDV'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['MEDV'] < (Q1 - 1.5 * IQR)) | (df['MEDV'] > (Q3 + 1.5 * IQR)))]

# Step 4: Explore Relationships - Plot 1: Scatter Plot (House Size vs. Price)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RM', y='MEDV', data=df, alpha=0.6)
plt.title('🏠 House Size (Rooms) vs. Price')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Median House Price (MEDV in $1000s)')
plt.savefig("visualizations/price_vs_size.png")
plt.show()

# Step 4b: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('📊 Feature Correlation Matrix')
plt.savefig("visualizations/correlation.png")
plt.show()

# Step 5: Build a Prediction Model
X = df[['RM', 'AGE', 'TAX', 'LSTAT']]  # Selected features
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n📈 Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualizing Actual vs Predicted Prices
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title("Actual vs. Predicted House Prices", fontsize=14)
plt.xlabel("Actual Prices", fontsize=12)
plt.ylabel("Predicted Prices", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

