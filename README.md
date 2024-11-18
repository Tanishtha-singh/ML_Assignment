1. *Dataset Exploration:*
   - Load the Iris dataset from `sklearn.datasets`.
   - Display the first five rows, the dataset’s shape, and summary statistics (mean, standard deviation, min/max values) for each feature.
solution -This Python program uses the **Iris dataset**, which is a built-in dataset in the `scikit-learn` library, and performs some basic exploratory data analysis. Below is a breakdown of each part of the code:

---

### 1. **Importing Libraries**
```python
from sklearn.datasets import load_iris
import pandas as pd
```
- **`sklearn.datasets.load_iris`**: This function loads the Iris dataset, a classic dataset for machine learning. It contains data about flowers, specifically their sepal and petal lengths and widths for three different species.
- **`pandas` (imported as `pd`)**: This library is used for data manipulation and analysis. It provides data structures like DataFrames, which are tables with labeled rows and columns.

---

### 2. **Loading the Dataset**
```python
data = load_iris()
```
- `load_iris()` returns a dictionary-like object containing the dataset. It includes:
  - **`data.data`**: The numerical data (features).
  - **`data.feature_names`**: The names of the features (e.g., "sepal length", "petal length").
  - **`data.target`**: The target values (species of iris flowers).
  - **`data.target_names`**: Names of the target classes (setosa, versicolor, virginica).

---

### 3. **Creating a DataFrame**
```python
df = pd.DataFrame(data.data, columns=data.feature_names)
```
- Converts the numerical data (`data.data`) into a **pandas DataFrame** with columns named after the feature names (`data.feature_names`).

---

### 4. **Displaying the First Five Rows**
```python
print("First five rows:\n", df.head())
```
- **`df.head()`**: Displays the first five rows of the DataFrame. It helps to get a quick look at the structure of the data.

---

### 5. **Displaying the Shape of the Dataset**
```python
print("\nDataset shape:", df.shape)
```
- **`df.shape`**: Returns a tuple `(rows, columns)` indicating the number of rows and columns in the dataset.

---

### 6. **Displaying Summary Statistics**
```python
print("\nSummary statistics:\n", df.describe())
```
- **`df.describe()`**: Generates summary statistics for the numerical columns. It includes:
  - **Count**: Number of non-missing values.
  - **Mean**: Average value.
  - **Std**: Standard deviation.
  - **Min**: Minimum value.
  - **25%**, **50%**, **75%**: Percentiles (quartiles).
  - **Max**: Maximum value.

---

### Example Output (Simplified)
1. **First Five Rows**:
   ```
   First five rows:
      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
   0               5.1              3.5               1.4              0.2
   1               4.9              3.0               1.4              0.2
   ...
   ```

2. **Dataset Shape**:
   ```
   Dataset shape: (150, 4)
   ```

3. **Summary Statistics**:
   ```
   Summary statistics:
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
   count         150.000000       150.000000        150.000000       150.000000
   mean            5.843333         3.057333          3.758000         1.199333
   std             0.828066         0.435866          1.765298         0.762238
   min             4.300000         2.000000          1.000000         0.100000
   ...
   ```

This program provides an initial understanding of the dataset, its structure, and basic statistical properties.


2. *Data Splitting:*
   - Split the Iris dataset into training and testing sets using an 80-20 split.
   - Print the number of samples in both the training and testing sets.

solution -This Python program uses the Iris dataset to split the data into training and testing sets. Here's a detailed explanation of the code:

---

### 1. **Importing Libraries**
```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
```
- **`train_test_split`**: A function from `sklearn.model_selection` that splits a dataset into training and testing subsets. This is a standard step in machine learning to evaluate model performance.
- **`load_iris`**: Loads the Iris dataset, which is commonly used in classification tasks.

---

### 2. **Loading the Dataset**
```python
data = load_iris()
X = data.data
y = data.target
```
- **`data.data (X)`**: The features (input variables) of the Iris dataset, which include measurements like sepal length, sepal width, petal length, and petal width. This is stored in `X`.
- **`data.target (y)`**: The target labels (output variable) representing the flower species (0 = setosa, 1 = versicolor, 2 = virginica). This is stored in `y`.

---

### 3. **Splitting the Data**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- **`train_test_split`**:
  - **`X` and `y`**: Inputs (features) and outputs (labels) of the dataset.
  - **`test_size=0.2`**: Specifies that 20% of the data will be used for testing, and 80% for training.
  - **`random_state=42`**: Ensures reproducibility of the split. Using the same `random_state` value will produce the same training/testing split every time.

- This function returns four subsets:
  - **`X_train`**: Training data features (80% of `X`).
  - **`X_test`**: Testing data features (20% of `X`).
  - **`y_train`**: Training data labels (80% of `y`).
  - **`y_test`**: Testing data labels (20% of `y`).

---

### 4. **Printing the Number of Samples**
```python
print("Number of training samples:", X_train.shape[0])
print("Number of testing samples:", X_test.shape[0])
```
- **`X_train.shape[0]`**: Returns the number of rows (samples) in the training data.
- **`X_test.shape[0]`**: Returns the number of rows (samples) in the testing data.

For the Iris dataset, which contains 150 samples, splitting it into 80% training and 20% testing results in:
- **Training samples**: 80% of 150 = 120 samples.
- **Testing samples**: 20% of 150 = 30 samples.

---

### Summary of Functionality
1. **Purpose**: The program splits the Iris dataset into training and testing subsets to train a machine learning model on one part (training set) and evaluate it on another (testing set).
2. **Output Example**:
   ```
   Number of training samples: 120
   Number of testing samples: 30
   ```


 3. *Linear Regression:*
   - Use a dataset with the features `YearsExperience` and `Salary`.
   - Fit a linear regression model to predict `Salary` based on `YearsExperience`
   - Fit a linear regression model to predict `Salary` based on `YearsExperience`. - Evaluate the model's performance using Mean Squared Error (MSE) on the test set

solution -  This program demonstrates how to perform linear regression on a small dataset using `scikit-learn`. It includes data splitting, model training, prediction, and evaluation using Mean Squared Error (MSE). Here’s a detailed explanation of each part of the code:

---

### 1. **Importing Libraries**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
```
- **`train_test_split`**: Splits data into training and testing sets.
- **`LinearRegression`**: A class for fitting linear regression models.
- **`mean_squared_error`**: A metric to evaluate the accuracy of a regression model. It calculates the average of squared differences between predicted and actual values.
- **`numpy` and `pandas`**: Libraries for numerical operations and data handling.

---

### 2. **Creating a Sample Dataset**
```python
data = pd.DataFrame({
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000]
})
```
- A small synthetic dataset is created using `pandas`. 
- **`YearsExperience`**: Independent variable (feature), representing work experience in years.
- **`Salary`**: Dependent variable (target), representing corresponding annual salaries.

---

### 3. **Defining Features and Target Variable**
```python
X = data[['YearsExperience']]
y = data['Salary']
```
- **`X`**: A DataFrame containing the independent variable (Years of Experience).
- **`y`**: A Series containing the target variable (Salary).

---

### 4. **Splitting the Data**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Splits the dataset into:
  - **Training set (80%)**: Used to train the model.
  - **Testing set (20%)**: Used to evaluate the model.
- **`random_state=42`** ensures the split is reproducible.

---

### 5. **Initializing and Training the Model**
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
- **`LinearRegression()`**: Creates a linear regression model instance.
- **`model.fit(X_train, y_train)`**: Trains the model by fitting a straight line to minimize the difference between the predicted and actual `y_train` values.

---

### 6. **Making Predictions**
```python
y_pred = model.predict(X_test)
```
- **`model.predict(X_test)`**: Predicts salary (`y_pred`) based on years of experience in the test set.

---

### 7. **Calculating Mean Squared Error**
```python
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on the test set:", mse)
```
- **`mean_squared_error(y_test, y_pred)`**: Calculates the Mean Squared Error, which quantifies the average squared difference between the actual and predicted values in the test set.
- A lower MSE indicates better model performance.

---

### **How the Linear Regression Works**
1. **Model Equation**: The regression model fits a straight-line equation of the form:
   \[
   \text{Salary} = m \times \text{YearsExperience} + b
   \]
   where \(m\) is the slope, and \(b\) is the intercept.
2. **Training**: The model learns \(m\) and \(b\) from the training data to minimize the error.
3. **Prediction**: The model uses the learned \(m\) and \(b\) to predict salaries for new inputs.

---

### **Output Example**
If executed, the program might print:
```
Mean Squared Error on the test set: 2500000.0
```
This indicates the model's average squared error on the test set.

### **Summary**
This code demonstrates a basic machine learning workflow:
1. **Load Data**: Define the input features (`X`) and target variable (`y`).
2. **Split Data**: Separate data into training and testing sets.
3. **Train Model**: Use training data to fit a linear regression model.
4. **Evaluate Model**: Use test data to compute prediction error (MSE).
