# Load and Preprocess the Dataset:

# Load and preprocess the dataset using Pandas. In this case, you will load the emu_legitimate_v1.csv and emu_malware_v1.csv files separately, and then concatenate them into a single DataFrame.





import pandas as pd

# Load malware dataset
malware_data = pd.read_csv("path_to_repo/data/emulator/emu_malware_v1.csv")

# Load legitimate dataset
legitimate_data = pd.read_csv("path_to_repo/data/emulator/emu_legitimate_v1.csv")

# Add a 'Malware' column to distinguish between malware (1) and legitimate (0)
malware_data['Malware'] = 1
legitimate_data['Malware'] = 0

# Combine malware and legitimate data
data = pd.concat([malware_data, legitimate_data], ignore_index=True)

# Make sure to adjust the file paths to match the structure of the cloned repository.

# Data Splitting:

# Split the combined dataset into features (X) and target labels (y), and then split them into training and testing sets.

from sklearn.model_selection import train_test_split

# Separate features (X) and target labels (y)
X = data.drop(columns=['Malware'])  # Remove the 'Malware' column
y = data['Malware']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Instantiate and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
print(classification_report(y_test, y_pred))
