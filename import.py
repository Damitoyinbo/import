
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/DAMITOYINBO/Documents/bank.csv")


df.shape
df.columns

df.head(10)

#Total count(loan) as a function of age
#Independent = age 
#dependent = loan(cnt)

# y = mx + c

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

# Fit and #predict

lm.fit(df["age"].reshape(-1,1),df["cnt"].reshape(-1,1))

lm.intercept_
lm.coef_
lm.predict(0.15)
df["age"].reshape(-1,1).shape

df.columns

lm.fit(df[["age","house"]],df["cnt"].reshape(-1,1))


lm.intercept_
lm.coef_
#y = m1x1 + m2x2 + c
plt.scatter(df["house"],df['cnt'])
