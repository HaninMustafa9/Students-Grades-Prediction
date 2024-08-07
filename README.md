# Students Grades Prediction

## Project Overview

This project aims to predict students' final grades (`G3`) using various features from a dataset. It involves data cleaning, encoding categorical variables, and scaling. A linear regression model is trained to predict final grades based on student-related features.

## Data Preprocessing

### Import Libraries

The following libraries are used:
- **pandas**: For data manipulation
- **numpy**: For numerical operations
- **matplotlib**: For visualization
- **seaborn**: For statistical visualization
- **sklearn**: For machine learning tasks

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
```

### Load the Data

The dataset is loaded from an external CSV file:

```python
Students_Data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/student-mat.csv")
```

### Exploratory Data Analysis (EDA)

1. **Describe the Data**

   ```python
   Students_Data.head()
   Students_Data.tail()
   Students_Data.shape
   Students_Data.info()
   Students_Data.describe()
   Students_Data.duplicated()
   ```

2. **Visualizations**

   Boxplots to visualize the distribution of features and detect outliers:

   ```python
   plt.figure(figsize = (15,10))
   sns.boxplot(data = Students_Data)
   plt.show()
   ```

### Data Cleaning

**Removing Outliers**

Outliers in the `absences` feature are removed based on the interquartile range (IQR):

```python
Q1_absences = Students_Data['absences'].quantile(0.25)
Q3_absences = Students_Data['absences'].quantile(0.75)
IQR_absences = Q3_absences - Q1_absences

lower_bound_absences = Q1_absences - 1.5 * IQR_absences
upper_bound_absences = Q3_absences + 1.5 * IQR_absences

Students_Data_no_outliers = Students_Data[(Students_Data['absences'] >= lower_bound_absences) & (Students_Data['absences'] <= upper_bound_absences)]
```

### Encoding and Scaling

**Encoding Categorical Variables**

```python
Students_Data_Encoded = pd.get_dummies(Students_Data_no_outliers, columns=['school', 'sex','famsize','address','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
```

**Scaling Features**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(Students_Data_Encoded.drop(columns=['G3']))
```

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Students_Data_Encoded['G3'], test_size=0.2, random_state=42)
```

## Model Training and Evaluation

**Training the Model**

```python
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

**Performance Metrics**

```python
print("Training Mean Squared Error:", mean_squared_error(y_train, y_train_pred))
print("Testing Mean Squared Error:", mean_squared_error(y_test, y_test_pred))
print("Training R^2 Score:", model.score(X_train, y_train))
print("Testing R^2 Score:", model.score(X_test, y_test))
```

## Conclusion

The project demonstrates the process of preparing data, training a linear regression model, and evaluating its performance to predict students' final grades based on various features.
