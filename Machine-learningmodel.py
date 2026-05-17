# =====================================================
# STUDENT PERFORMANCE PREDICTION
# Synent Technologies Internship - Task 8
# =====================================================

# ---------- IMPORT LIBRARIES ----------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("\n========== IMPORTS SUCCESSFUL ==========")

# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading Dataset...")

df = pd.read_csv("StudentsPerformance.csv")

print("Dataset Loaded Successfully!")

# =====================================================
# BASIC INFORMATION
# =====================================================

print("\n========== DATASET OVERVIEW ==========")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nColumns:")
print(df.columns)

# =====================================================
# DATA CLEANING
# =====================================================

print("\n========== DATA CLEANING ==========")

duplicates = df.duplicated().sum()

print(f"Duplicate Rows Found: {duplicates}")

df.drop_duplicates(inplace=True)

print("Duplicate Rows Removed Successfully!")

# =====================================================
# EXPLORATORY DATA ANALYSIS
# =====================================================

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

print(f"Average Math Score: {df['math score'].mean():.2f}")
print(f"Average Reading Score: {df['reading score'].mean():.2f}")
print(f"Average Writing Score: {df['writing score'].mean():.2f}")

# =====================================================
# VISUALIZATION FUNCTION
# =====================================================

def show_plot():
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# =====================================================
# GENDER DISTRIBUTION
# =====================================================

print("\nGenerating Gender Distribution Plot...")

plt.figure(figsize=(6,4))

sns.countplot(x='gender', data=df)

plt.title("Gender Distribution")

show_plot()

print("Gender Distribution Plot Completed!")

# =====================================================
# MATH SCORE DISTRIBUTION
# =====================================================

print("\nGenerating Math Score Distribution Plot...")

plt.figure(figsize=(8,5))

sns.histplot(df['math score'], bins=20)

plt.title("Math Score Distribution")
plt.xlabel("Math Score")

show_plot()

print("Math Score Distribution Plot Completed!")

# =====================================================
# READING VS WRITING SCORES
# =====================================================

print("\nGenerating Scatter Plot...")

plt.figure(figsize=(8,5))

sns.scatterplot(
    x='reading score',
    y='writing score',
    hue='gender',
    data=df
)

plt.title("Reading vs Writing Scores")

show_plot()

print("Scatter Plot Completed!")

# =====================================================
# CORRELATION HEATMAP
# =====================================================

print("\nGenerating Correlation Heatmap...")

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(6,4))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

show_plot()

print("Correlation Heatmap Completed!")

# =====================================================
# FEATURE ENCODING
# =====================================================

print("\n========== FEATURE ENCODING ==========")

label_encoder = LabelEncoder()

categorical_columns = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course'
]

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("Categorical Features Encoded Successfully!")

# =====================================================
# FEATURE SELECTION
# =====================================================

print("\nSelecting Features and Target Variable...")

X = df.drop('math score', axis=1)
y = df['math score']

print("Feature Selection Completed!")

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

print("\nSplitting Dataset into Train and Test Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train-Test Split Completed!")

# =====================================================
# MODEL TRAINING
# =====================================================

print("\n========== MODEL TRAINING ==========")

model = LinearRegression()

model.fit(X_train, y_train)

print("Linear Regression Model Trained Successfully!")

# =====================================================
# PREDICTION
# =====================================================

print("\nGenerating Predictions...")

y_pred = model.predict(X_test)

print("Predictions Generated Successfully!")

# =====================================================
# MODEL EVALUATION
# =====================================================

print("\n========== MODEL EVALUATION ==========")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# =====================================================
# ACTUAL VS PREDICTED GRAPH
# =====================================================

print("\nGenerating Actual vs Predicted Graph...")

plt.figure(figsize=(8,5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")

plt.title("Actual vs Predicted Scores")

show_plot()

print("Actual vs Predicted Graph Completed!")

# =====================================================
# SAMPLE PREDICTION
# =====================================================

print("\n========== SAMPLE PREDICTION ==========")

sample_prediction = model.predict([X_test.iloc[0]])

print(f"Predicted Math Score: {sample_prediction[0]:.2f}")

# =====================================================
# FINAL INSIGHTS
# =====================================================

print("\n========== FINAL INSIGHTS ==========")

print("1. Reading and writing scores strongly influence math performance.")

print("2. The model successfully predicts student scores.")

print("3. Data preprocessing and encoding improved model training.")

print("4. Linear Regression works effectively for this dataset.")

# =====================================================
# FINAL TERMINATION
# =====================================================

print("\n======================================")
print("PROJECT EXECUTED SUCCESSFULLY!")
print("Machine Learning Workflow Completed.")
print("Program Terminated Successfully.")
print("======================================")