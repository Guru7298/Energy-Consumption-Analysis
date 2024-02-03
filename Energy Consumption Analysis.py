import pandas as pd

data = pd.read_csv('energy_consumption_data.csv')

data.dropna(inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns


print(data.describe())


corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


sns.histplot(data['energy_consumption'], kde=True)
plt.title('Energy Consumption Distribution')
plt.xlabel('Energy Consumption')
plt.ylabel('Frequency')
plt.show()

data.to_csv('cleaned_data.csv', index=False)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[['feature1', 'feature2', ...]]
y = data['energy_consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)

