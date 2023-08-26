import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balancing the Dataset (using SMOTE)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Visualizing the Balanced Classes
plt.figure(figsize=(6, 4))
sns.countplot(y_train)
plt.title('Class Distribution')
plt.xlabel('Cardiovascular disease')
plt.ylabel('Count')
plt.show()

# Summary Statistics
print("Summary Statistics:")
print(data.describe())

# Visualizing the Distribution of the Target Variable
plt.figure(figsize=(6, 4))
sns.countplot(data['cardio'])
plt.title('Distribution of Cardiovascular Disease')
plt.xlabel('Cardiovascular disease')
plt.ylabel('Count')
plt.show()

# Exploring the Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Analyzing the Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Examining Categorical Variables
categorical_features = ['cholesterol', 'gluc', 'smoke', 'alco', 'active']
for feature in categorical_features:
    plt.figure(figsize=(6, 4))
    sns.countplot(data[feature], hue=data['cardio'])
    plt.title(f'{feature.capitalize()} Distribution')
    plt.xlabel(f'{feature.capitalize()}')
    plt.ylabel('Count')
    plt.legend(title='Cardiovascular disease')
    plt.show()

# Comparing Height Distribution across Gender
plt.figure(figsize=(8, 6))
sns.violinplot(x='gender_1', y='height', data=data)
plt.title('Height Distribution across Gender')
plt.xlabel('Gender')
plt.ylabel('Height')
plt.show()

# Creating a New Feature (BMI) and Comparing it between Healthy and Ill Individuals
data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

plt.figure(figsize=(6, 4))
sns.boxplot(x='cardio', y='bmi', data=data)
plt.title('BMI Distribution')
plt.xlabel('Cardiovascular disease')
plt.ylabel('BMI')
plt.show()


# Initialize classifiers with adjusted hyperparameters
classifiers = {
    "Decision Tree": DecisionTreeClassifier(criterion='gini', random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Support Vector Machine": SVC(random_state=42, kernel='linear', probability=True),  # Enable probability estimation
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, subsample=0.8)
}
# Iterate through classifiers
for name, classifier in classifiers.items():
    print("=" * 50)
    print("Classifier:", name)

    start_time = time.time()  # Start timer

# Initialize a dictionary to store the results
results = []

    # Iterate through classifiers
for name, classifier in classifiers.items():
    print("=" * 50)
    print("Classifier:", name)

    start_time = time.time()  # Start timer
    
    try:
        # Fit the classifier to the training data
        classifier.fit(X_train, y_train)
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print("Training time:", elapsed_time, "seconds")

        # Predict on the test data
        y_pred = classifier.predict(X_test)
        y_pred_prob = classifier.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 1
        
        # Calculate and display Performance Metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        confusion_mat = confusion_matrix(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_prob)  # Calculate AUC-ROC
        
        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_rep)
        print("Confusion Matrix:\n", confusion_mat)
        print("AUC-ROC:", auc_roc)
        
        # Interpretability: Display Feature Importance for applicable algorithms
        if name in ["Decision Tree", "Random Forest", "Gradient Boosting"]:
            feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": classifier.feature_importances_})
            feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
            print("\nFeature Importance for", name, "on Cardiovascular Disease Dataset:")
            print(feature_importance)
        elif name == "Logistic Regression":
            feature_coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": classifier.coef_[0]})
            feature_coefficients = feature_coefficients.sort_values(by="Coefficient", ascending=False)
            print("\nFeature Coefficients for Logistic Regression:")
            print(feature_coefficients)

    # Store the results in the dictionary
        result = {
            "Classifier": name,
            "Training time": elapsed_time,
            "Accuracy": accuracy,
            "AUC-ROC": auc_roc
            # Add more metrics if needed
        }
        results.append(result)

    except Exception as e:
        print("Error during fitting or evaluation:", e)

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Print the comparison table
print("\nComparison of Classifier Performance:")
print(results_df)

# Plot AUC-ROC curves
plt.figure(figsize=(10, 6))
for name, classifier in classifiers.items():
    if hasattr(classifier, "predict_proba"):  # Check if classifier supports predict_proba
        y_pred_prob = classifier.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_prob)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_roc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()