import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')
dataset = pd.read_csv('churn.csv')
dataset
# EDA

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
dataset[['Geography_France','Geography_Germany',
         'Geography_Spain','EstimatedSalary','Exited']]

# Feature Importance

plt.figure(figsize=(10, 4))
correl_matrix = dataset.corr().round(2)
sns.heatmap(data=correl_matrix, annot=True)
plt.show()
X = dataset[['Age','Geography_Germany','IsActiveMember','Balance','Gender']]
y = dataset['Exited']

X
y

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                        random_state=13)
X_train
# Scaling

scaler = MinMaxScaler()
X_train[['Age','Balance']] = scaler.fit_transform(
    X_train[['Age','Balance']])
X_test[['Age','Balance']] = scaler.transform(
    X_test[['Age','Balance']])
X_train
svc_lin1 = SVC(kernel='linear',random_state=13,C=1)
svc_lin1.fit(X_train,y_train)
y_pred_lin1 = svc_lin1.predict(X_test)

print('Model performance on Training Set : \n')
print(classification_report(y_train,svc_lin1.predict(X_train)))
print()
print()
print('Model performance on Test Set : \n')
print(classification_report(y_test,y_pred_lin1))
print()
print()
svc_lin2 = SVC(kernel='linear',random_state=13,C=10)
svc_lin2.fit(X_train,y_train)
y_pred_lin2 = svc_lin2.predict(X_test)

print('Model performance on Training Set : \n')
print(classification_report(y_train,svc_lin2.predict(X_train)))
print()
print()
print('Model performance on Test Set : \n')
print(classification_report(y_test,y_pred_lin2))
print()
print()
svc_lin3 = SVC(kernel='linear',random_state=13,C=100)
svc_lin3.fit(X_train,y_train)
y_pred_lin3 = svc_lin3.predict(X_test)

print('Model performance on Training Set : \n')
print(classification_report(y_train,svc_lin3.predict(X_train)))
print()
print()
print('Model performance on Test Set : \n')
print(classification_report(y_test,y_pred_lin3))
print()
print()
