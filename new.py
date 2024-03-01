import pandas as pd

# Replace with the actual path to your CSV file
melbourne_file_path = "D:\Machine Learning\kaggle\melbourne_data\melbourne_data.csv"

try:
  # Read the data using the correct path
  melbourne_data = pd.read_csv(melbourne_file_path)

  # Print the summary of the data
  print(melbourne_data.describe())
except FileNotFoundError:
  print("Error: File not found at", melbourne_file_path)

# get count of built house
grouped = melbourne_data[["Address","YearBuilt"]].groupby("YearBuilt").count()

grouped.sort_values("Address",ascending=False)

print(grouped)

print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0)
print(melbourne_data)

y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

print('x.head() x.tail()')

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(x, y)

print("Making predictions for the following 5 houses:")
print(x.head())
print("The predictions are")
print(melbourne_model.predict(x.head()))

from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(x)
mean_absolute_error(y, predicted_home_prices)

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_x, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))