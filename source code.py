import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('sensor_data.csv')
# Display the first few rows of the dataset
print(df.head())
# Check for missing values
print(df.isnull().sum())
print(df['Event'].value_counts())
# Plot the distribution of required oxygen levels
plt.figure(figsize=(6, 4))
sns.histplot(df['Event'], bins=30, kde=True)
plt.title('Detect Natural Disasters')
plt.xlabel('Noraml Or Natural Disasters')
plt.ylabel('sensor datas')
plt.show()
# Step 5: Correlation Analysis
print("\nCorrelation Matrix:")
correlation_matrix = df.corr()
print(correlation_matrix)

# Visualize the correlation matrix with a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Sensor Data')
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size for the plots
plt.figure(figsize=(15, 6))

# Plot the distribution of Temperature
plt.subplot(1, 6, 1)
sns.histplot(df['Accel_X'], bins=30, kde=True)
plt.title('Accel_X')

# Plot the distribution of Temperature
plt.subplot(1, 6, 2)
sns.histplot(df['Accel_Y'], bins=30, kde=True)
plt.title('Accel_Y')

# Plot the distribution of Temperature
plt.subplot(1, 6, 3)
sns.histplot(df['Accel_Z'], bins=30, kde=True)
plt.title('Accel_Z')

# Plot the distribution of Pulse Rate
plt.subplot(1, 6, 4)
sns.histplot(df['Flood_Level'], bins=30, kde=True)
plt.title('Flood_Level')

# Plot the distribution of Oximeter
plt.subplot(1, 6, 5)
sns.histplot(df['Fire_Status'], bins=30, kde=True)
plt.title('Fire_Status')

# Plot the distribution of ECG
plt.subplot(1, 6, 6)
sns.histplot(df['Dam_Level'], bins=30, kde=True)
plt.title('Dam_Level')



# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
# Separate features (sensor values) and the target (Label)
X = df.drop(columns=['Event'])  # Features: sensor values
y = df['Event']  # Target: earthquake, flood, or normal
# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

# Scale the features to standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
accuracy = rf_model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

# Evaluate the model
accuracy = gb_model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
# Save the trained model
import pickle

# Save the Random Forest model to a file
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

# Save the scaler used for feature scaling
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
          import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Use Classifier instead of Regressor
from sklearn.ensemble import GradientBoostingClassifier  # Use Classifier instead of Regressor

# Load the saved models
with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Function to make predictions using the trained model
def predict_event(accel_x, accel_y, accel_z, flood_level, fire_status, dam_level):
    # Prepare the input data (reshape into a 2D array for prediction)
    input_data = np.array([[accel_x, accel_y, accel_z, flood_level, fire_status, dam_level]])

    # Scale the input data using the same scaler used for training
    input_data_scaled = scaler.transform(input_data)

    # Make prediction with the Random Forest model
    prediction_encoded = rf_model.predict(input_data_scaled)

    # Decode the prediction back to the original event label
    label_encoder = LabelEncoder()
    # Fit the encoder on the labels used in the training
    label_encoder.fit(['Earthquake', 'Flood', 'Fire', 'Normal'])  # Update with the labels used in your training
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    return prediction_label[0]

# Function to take user inputs and predict the result
def get_user_input_and_predict():
    # Get user inputs for the sensor values
    try:
        accel_x = float(input("Enter the value for Accel_X: "))
        accel_y = float(input("Enter the value for Accel_Y: "))
        accel_z = float(input("Enter the value for Accel_Z: "))
        flood_level = float(input("Enter the value for Flood Level: "))
        fire_status = int(input("Enter the value for Fire Status (1 for Fire, 0 for No Fire): "))
        dam_level = float(input("Enter the value for Dam Level: "))

        # Predict the event
        predicted_event = predict_event(accel_x, accel_y, accel_z, flood_level, fire_status, dam_level)
        print(f"The predicted event is: {predicted_event}")
    except ValueError:
        print("Invalid input! Please enter valid numeric values.")

# Run the function to get user input and make predictions
get_user_input_and_predict()
