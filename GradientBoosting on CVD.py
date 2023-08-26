import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import time

# Load the dataset
data = pd.read_csv("C:/Users/divya/OneDrive/Desktop/Project 2023/Dataset/Main/Cardiovascular Disease dataset.csv")

# Handling Missing Values
data.dropna(inplace=True)  # Remove rows with missing values

# Dealing with Duplicate Data
data.drop_duplicates(inplace=True)

# Feature Selection
selected_features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
data = data[selected_features]

# Encoding Categorical Variables
data = pd.get_dummies(data, columns=['gender'], prefix='gender')

# Standardization or Normalization (using Min-Max scaling)
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
data[numerical_features] = (data[numerical_features] - data[numerical_features].min()) / (data[numerical_features].max() - data[numerical_features].min())

# Handling Outliers (using boxplots)
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numerical_features])
plt.title('Boxplot of Numerical Features')
plt.show()

# Splitting the Data
X = data.drop('cardio', axis=1)
y = data['cardio']

# Feature Selection
selected_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'gender_1', 'gender_2']
X = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balancing the Dataset (using SMOTE)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Initialize Gradient Boosting classifier with adjusted hyperparameters
gradient_boosting = GradientBoostingClassifier(random_state=42, subsample=0.8)  # Adjust subsample

print("=" * 50)
print("Classifier: Gradient Boosting")

start_time = time.time()  # Start timer

try:
    # Fit the Gradient Boosting classifier to the training data
    gradient_boosting.fit(X_train, y_train)
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    print("Training time:", elapsed_time, "seconds")

    # Display Algorithm Configuration
    print("\nAlgorithm Configuration:")
    print("Subsample:", gradient_boosting.subsample)

    # Predict on the test data
    y_pred = gradient_boosting.predict(X_test)

    # Calculate and display Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", confusion_mat)

    # Feature Importance
    feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": gradient_boosting.feature_importances_})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    print("\nFeature Importance for Gradient Boosting:")
    print(feature_importance)

except Exception as e:
    print("Error during fitting or evaluation:", e)
