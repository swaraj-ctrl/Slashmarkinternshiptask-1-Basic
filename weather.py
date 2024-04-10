import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error ,accuracy_score


data=pd.read_csv('D:/Users/varun/Downloads/basic/basic/weather.csv')
mean_value=data['WindSpeed9am'].mean()
data['WindSpeed9am'].fillna(value=mean_value, inplace=True)
data.info()
new_data = data.dropna()
print(new_data.head(2))
new_data.info()
seaborn.pairplot(new_data, hue ='Cloud9am')
#plot.show()
sns.histplot(data['MaxTemp'], kde=True)
plot.show()
#sns.pairplot(new_data, diag_kind='kde')
#plot.show()

data['temperature_squared'] = data['MaxTemp'] ** 2

correlation = data['MaxTemp'].corr(data['Humidity9am'])
print(f"Correlation between temperature and humidity: {correlation}")


sns.boxplot(x='Rainfall', data=data)
plot.show()
sns.scatterplot(x='MaxTemp', y='RainTomorrow', data=data)
plot.title('Relationship between MaxTemp and RainTomorrow')
plot.xlabel('MaxTemp')
plot.ylabel('RainTomorrow')
plot.show()


sns.scatterplot(x='Rainfall', y='RainTomorrow', data=new_data)
plot.title('Relationship between Rainfall and RainTomorrow')
plot.xlabel('Rainfall')
plot.ylabel('RainTomorrow')
plot.show()

sns.scatterplot(x='Humidity9am', y='RainTomorrow', data=new_data)
plot.title('Relationship between Humidity9am and RainTomorrow')
plot.xlabel('Humidity9am')
plot.ylabel('RainTomorrow')
plot.show()

sns.scatterplot(x='Humidity3pm', y='RainTomorrow', data=new_data)
plot.title('Relationship between Humidity9am and RainTomorrow')
plot.xlabel('Humidity3pm')
plot.ylabel('RainTomorrow')
plot.show()
print(new_data.info())
label_encoder = preprocessing.LabelEncoder() 
  

new_data['RainToday']= label_encoder.fit_transform(new_data['RainToday'])
print(new_data['RainToday'].unique())
new_data['RainTomorrow']= label_encoder.fit_transform(new_data['RainTomorrow']) 
print(new_data['RainTomorrow'].unique()) 
X = new_data.drop(new_data.columns[[5, 7, 8]], axis=1)
y = new_data['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
Acc = accuracy_score(y_test, y_pred)
print("Accuracy Score:", Acc)