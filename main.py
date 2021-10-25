import pandas as pd
import csv
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

temp = df["Temperature"].tolist()
melted = df["Melted"].tolist()

fig = px.scatter(x=temp, y=melted)
fig.show()

#adding the line of regression (best fit)
temparray = np.array(temp)
meltedarray = np.array(melted)

#slope and intercept
m, c = np.polyfit(temparray, meltedarray, 1)
y = []

for x in temparray:
    yvalue = (m*x) + c
    y.append(yvalue)

fig2 = px.scatter(x=temparray, y=meltedarray)
fig2.update_layout(shapes = [
    dict(
        type = "line",
        y0 = min(y),
        y1 = max(y),
        x0 = min(temparray),
        x1 = max(temparray),
    )
])
fig2.show()

#reshape the list
X = np.reshape(temp, (len(temp), 1))
Y = np.reshape(melted, (len(melted), 1))

lr = LogisticRegression()
lr.fit(X, Y)
plt.figure()
plt.scatter(X.ravel(), Y, color = "black", zorder = 20)

def model(x):
    return 1/(1+np.exp(-x))

#using the line formula
xtest = np.linspace(0, 5000, 10000)
chances = model((xtest*lr.coef_) + lr.intercept_).ravel()

#plotting the graph

plt.plot(xtest, chances, color = "red", linewidth = 3)
plt.axhline(y = 0, color = "k", linestyle = "-")
plt.axhline(y = 1, color = "k", linestyle = "-")
plt.axhline(y = 0.5, color = "b", linestyle = "-")
plt.axvline(x = xtest[6843], color = "b", linestyle = "--")

plt.ylabel("Y")
plt.xlabel("X")
plt.xlim(3400, 3450)
plt.show()

temperature = float(input("Enter the temperature: "))
chancesmelt = model((temperature*lr.coef_) + lr.intercept_).ravel()[0]
if chancesmelt <= 0.01:
    print("Tungsten will not melt")
elif chancesmelt >= 1:
    print("Tungsten will melt")
elif chancesmelt <= 0.5:
    print("Tungsten might not melt")
else:
    print("Tungsten might melt")
