# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # SVM model
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the cleaned data (replace 'your_cleaned_data.csv' with your actual file)
df = pd.read_csv("your_cleaned_data.csv")

# Step 2: Data Preprocessing

# Assume the target column is 'threat' (0 for no threat, 1 for threat)
# and all other columns are features

# X is the feature matrix (input data)
# y is the target vector (labels)

X = df.drop(columns=['threat'])  # Features (all columns except 'threat')
y = df['threat']  # Target variable (0 or 1 indicating threat)

# Optional: If you have categorical features, encode them (e.g., using OneHotEncoder or LabelEncoder)
# Here we assume all features are numerical for simplicity.

# Step 3: Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')  # You can choose other kernels like 'rbf', 'poly', etc.
svm_model.fit(X_train_scaled, y_train)

# Step 6: Make predictions and evaluate the SVM model
svm_predictions = svm_model.predict(X_test_scaled)

# Evaluate the SVM model
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

# Step 7: Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 8: Make predictions and evaluate the Random Forest model
rf_predictions = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))
