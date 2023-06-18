#Example 1-1.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

url = "https://github.com/ageron/data/raw/main/lifesat/lifesat.csv"

data = pd.read_csv(url)

X = data[["GDP per capita (USD)"]].values
y = data[["Life satisfaction"]].values

data.plot(
    kind="scatter",
    x="GDP per capita (USD)",
    y="Life satisfaction",
    )

plt.show()

model = LinearRegression()

model.fit(X, y)

poland_gdp_per_capita = 17_999.91

X_new = [[poland_gdp_per_capita]]

print(model.predict(X_new))