# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start and import the required Python libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

2. Load the Dataset containing student details such as CGPA, IQ, communication skills, technical skills, and placement status.

3. Preprocess the Dataset by handling missing values, removing duplicates, encoding categorical variables, and selecting relevant features.

4. Split the Dataset into training and testing data (80% for training and 20% for testing).

5. Build and Train the Model by applying feature scaling and fitting the Logistic Regression algorithm using the training dataset.

6. Predict the Placement Status of students using the trained Logistic Regression model.

7. Evaluate the Model using accuracy, precision, recall, F1-score, and confusion matrix.

8. Display the Results by showing whether the student is Placed or Not Placed, and stop the program.

## Program:
```

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Placement_Data.csv")

print("First 5 rows of the dataset:")
print(data.head())

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())

print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())

X = data1.iloc[:, :-1]
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

lr = LogisticRegression(solver="liblinear")

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])

```

## Output:
<img width="791" height="895" alt="image" src="https://github.com/user-attachments/assets/36ee2cc7-1758-492f-9366-4064a77d9f1f" />
<img width="805" height="963" alt="image" src="https://github.com/user-attachments/assets/a1853ae2-598d-49e7-8cfd-43c8376e0e04" />
<img width="778" height="933" alt="image" src="https://github.com/user-attachments/assets/e070721d-1fac-4932-9dd1-4aff3d0f5cef" />
<img width="850" height="978" alt="image" src="https://github.com/user-attachments/assets/9727b24d-5e27-4cac-889b-246c0b4b8e93" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
