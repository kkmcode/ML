# Importing libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_csv('churn.csv')[0:201]
dataset
print('Columns : ',list(dataset.columns))
print()
print('Number of missing values : ',dataset.isnull().sum().sum())
print()
dataset.describe()
# Data Analysis

print(dataset['Geography'].unique())
print(dataset['Gender'].unique())

# Data Preprocessing

le = LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])

dataset = pd.get_dummies(dataset,columns = ['Geography'])
dataset
dataset[['Geography_France','Geography_Germany','Geography_Spain','EstimatedSalary','Exited']]
# Feature Importance

plt.figure(figsize=(10, 4))
correl_matrix = dataset.corr().round(2)
sns.heatmap(data=correl_matrix, annot=True)
plt.show()
X = dataset[['Age','Geography_France','IsActiveMember','HasCrCard','Gender']]
y = dataset['Exited']

X
y

# Train test split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y,test_size=0.1,
 random_state=13)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X,y,test_size=0.2,
 random_state=13)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X,y,test_size=0.3,
 random_state=13)
X_train1
y_train1
# Decision Tree Classifier for 80-20 Split
dtree = DecisionTreeClassifier(random_state=13)
dtree.fit(X_train2,y_train2)
y_pred_train2 = dtree.predict(X_train2)
y_pred_test2 = dtree.predict(X_test2)
print('80-20 Model performance on Training Set : \n')
print(classification_report(y_train2,y_pred_train2))
print()
print()
print('80-20 Model performance on Test Set : \n')
print(classification_report(y_test2,y_pred_test2))
print()
print()
X_ip = list(map(int,
 input("Enter Age, isActiveMember, HasCrCard, Gender, Geography_France : ")
 .split()))[:5]
print('Predicted class : ',dtree.predict([X_ip])[0])
print()
print()
print('Result : ')
pd.DataFrame({'Actual':y_test2,'Predicted':y_pred_test2})
# Visualize the Tree
plt.figure(figsize=(40,40))
plot_tree(dtree, feature_names = X_train2.columns, filled=True, fontsize=20,
 rounded = True)
plt.show()