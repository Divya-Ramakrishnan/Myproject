import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv("C:/Users/divya/OneDrive/Desktop/Project 2023/Dataset/Main/heart_failure_clinical_records_dataset.csv")

# Handling Missing Values
data.dropna(inplace=True)  # Remove rows with missing values

# Dealing with Duplicate Data
data.drop_duplicates(inplace=True)

# Feature Selection
selected_features = ['age', 'anaemia', 'high_blood_pressure', 'creatinine_phosphokinase',
                     'diabetes', 'ejection_fraction', 'platelets', 'sex',
                     'serum_creatinine', 'serum_sodium', 'smoking', 'time', 'DEATH_EVENT']
data = data[selected_features]

# Encoding Categorical Variables
categorical_features = ['anaemia', 'high_blood_pressure', 'diabetes', 'sex', 'smoking']
encoder = OneHotEncoder(sparse=False)
encoded_features = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
encoded_features.columns = encoder.get_feature_names(categorical_features)
data = pd.concat([data.drop(categorical_features, axis=1), encoded_features], axis=1)

# Standardization or Normalization
numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
                      'serum_creatinine', 'serum_sodium', 'time']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Handling Outliers 

numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
                      'serum_creatinine', 'serum_sodium', 'time']
z_scores = stats.zscore(data[numerical_features])
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)  # Adjust the threshold as needed
data = data[filtered_entries]


# Balancing the Dataset (using SMOTE)
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

## Distribution of the Target Variable
plt.figure(figsize=(6, 4))
sns.countplot(data['DEATH_EVENT'])
plt.title('Distribution of Death Event')
plt.xlabel('Death Event')
plt.ylabel('Count')
plt.show()

# Relationship between Age and Death Event
plt.figure(figsize=(8, 6))
sns.boxplot(x='DEATH_EVENT', y='age', data=data)
plt.title('Age vs. Death Event')
plt.xlabel('Death Event')
plt.ylabel('Age')
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairwise Relationships
sns.pairplot(data, vars=['age', 'ejection_fraction', 'serum_creatinine', 'time'], hue='DEATH_EVENT')
plt.title('Pairwise Relationships')
plt.show()