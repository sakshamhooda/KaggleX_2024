import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to check for missing values and handle them
def handle_missing_values(df):
    imputer = SimpleImputer(strategy='median')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

# Load the new dataset
file_path_new = 'train.csv'
data_new = pd.read_csv(file_path_new)

# Separate the features and target
X_new = data_new.drop(columns='price')
y_new = data_new['price']

# Apply ordinal encoding to categorical features
categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
ordinal_encoder_new = OrdinalEncoder()
X_new[categorical_features] = ordinal_encoder_new.fit_transform(X_new[categorical_features])

# Handle missing values in numerical columns with the median
X_new = handle_missing_values(X_new)

# Split the data into training and testing sets
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# Train and evaluate Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_new, y_train_new)
y_pred_lr = lr_model.predict(X_test_new)
mae_lr = mean_absolute_error(y_test_new, y_pred_lr)
mse_lr = mean_squared_error(y_test_new, y_pred_lr)
r2_lr = r2_score(y_test_new, y_pred_lr)

# Train and evaluate Random Forest Regressor model
rf_model_new = RandomForestRegressor(random_state=42)
rf_model_new.fit(X_train_new, y_train_new)
y_pred_rf_new = rf_model_new.predict(X_test_new)
mae_rf_new = mean_absolute_error(y_test_new, y_pred_rf_new)
mse_rf_new = mean_squared_error(y_test_new, y_pred_rf_new)
r2_rf_new = r2_score(y_test_new, y_pred_rf_new)

# Prepare the results
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest Regressor'],
    'MAE': [mae_lr, mae_rf_new],
    'MSE': [mse_lr, mse_rf_new],
    'R2': [r2_lr, r2_rf_new]
})

print(results)
