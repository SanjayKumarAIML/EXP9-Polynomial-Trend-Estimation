
# <p align="center">Polynomial Trend Estimation</p>

## AIM :
To estimate polynomial trend using python.

## ALGORITHM :

### STEP 1: 
Create a function to estimate linear trend.
### STEP 2: 
Using the created function find out the linear trend for some statistical data.
### STEP 3: 
Import necessary libraries to estimate polynomial trend.
### STEP 4: 
Read the csv file.
### STEP 5: 
Assign values for x and y from the dataset.
### STEP 6: 
Fit them with the help of LinearRegression function.
### STEP 7: 
Display the data in graph using matplotlib.
### STEP 8: 
Estimate the trends.

## PROGRAM :
Developed By : **Sanjay Kumar S S**
</br>
Register No. : **212221240048**
### Linear trend estimation:
```py
# Function to calculate b
def calculateB(x, y, n):
    sx = sum(x)
    sy = sum(y) 
    sxsy = 0
    sx2 = 0
    for i in range(n):
        sxsy += x[i] * y[i]
        sx2 += x[i] * x[i]
    b = (n * sxsy - sx * sy)/(n * sx2 - sx * sx)
    return b

# Function to find the least regression line
def leastRegLine(X,Y,n):
    b = calculateB(X, Y, n)
    meanX = int(sum(X)/n)
    meanY = int(sum(Y)/n)
    a = meanY - b * meanX
    print("Linear Trend:")
    print("Y = ", '%.3f'%a, " + ", '%.3f'%b, "*X", sep="")

# Driver code
# Statistical data 
X = [95, 85, 80, 70, 60 ]
Y = [90, 80, 70, 65, 60 ]
n = len(X)
leastRegLine(X, Y, n)
```
### Polynomial Regression:
```py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas = pd.read_csv('data.csv')
datas

X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values
X = datas.iloc[:, 1:2].values
y = datas.iloc[:, 2].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures 
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)
lin2 = LinearRegression()
lin2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color='blue')
plt.plot(X, lin.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='blue')
plt.plot(X, lin2.predict(poly.fit_transform(X)),color='red')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.show()

# Predicting a new result with Linear Regression after converting predict variable to 2D array
pred = 110.0
predarray = np.array([[pred]])
lin.predict(predarray)

# Predicting a new result with Polynomial Regression after converting predict variable to 2D array
pred2 = 110.0
pred2array = np.array([[pred2]])
lin2.predict(poly.fit_transform(pred2array))
```
## OUTPUT :
### Linear Trend:
![image](https://github.com/Jovita08/Polynomial-Trend-Estimation/assets/94174503/c56ce583-ca41-474a-b6a2-8a44e6a98905)
### Linear Regression:
![image](https://github.com/Jovita08/Polynomial-Trend-Estimation/assets/94174503/fb28ddf4-325a-477b-90d1-e6fd608a87f8)
### Polynomial Regression:
![image](https://github.com/Jovita08/Polynomial-Trend-Estimation/assets/94174503/77000b21-2a48-443e-aa40-262606f5f110)
### After conversion of predicted variable to 2D array:
#### Linear Regression:
![image](https://github.com/Jovita08/Polynomial-Trend-Estimation/assets/94174503/7d186a42-8e5d-4e25-bac3-8a363af46795)
#### Polynomial Regression:
![image](https://github.com/Jovita08/Polynomial-Trend-Estimation/assets/94174503/5cd487d8-911d-4a3e-8f1d-d4f37355fed0)

## RESULT :
Thus,the program successfully estimated the polynomial trend.
