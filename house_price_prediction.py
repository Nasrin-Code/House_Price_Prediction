from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing(as_frame=True)

print(housing.data.head())
print(housing.target.head())

print(housing.data.shape)
print(housing.data.info())
print(housing.data.describe())

print(housing.data.isnull().sum())

X = housing.data
y = housing.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[:10])

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error", mse)