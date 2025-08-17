import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("advertising.csv")

print("First five rows of the dataset:")
print(df.head())

features = ["TV", "Radio", "Newspaper"]
target = "Sales"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")


new_data = pd.DataFrame({
    "TV": [150.0],
    "Radio": [30.0],
    "Newspaper": [20.0]
})
predicted_sales = model.predict(new_data)
print(f"\nPredicted Sales for TV=$150, Radio=$30, Newspaper=$20: {predicted_sales[0]:.2f}")

