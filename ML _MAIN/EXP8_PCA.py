# Importing Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
warnings.filterwarnings('ignore')
# Reading dataset

dataset = pd.read_csv('bank.csv')
dataset
# Correlation Matrix

correl = dataset.corr()
sns.heatmap(correl,annot=True)
plt.show()
X = dataset[['variance','skewness','curtosis']]
y = dataset['class']
X
# Splitting data into training and test set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                    random_state=13)
X_train
# Standardization

sc = StandardScaler()
X_train[['variance','skewness','curtosis']] =  sc.fit_transform(X_train)
X_test[['variance','skewness','curtosis']]  = sc.transform(X_test)
X_train
X_test

# 3D plot of features

fig = plt.figure()
ax = plt.axes(projection ='3d')

ax.scatter(X_train['variance'],X_train['skewness'],X_train['curtosis'])
ax.set_title('3D plot of Training set')
plt.show()
# PCA

pca = PCA(n_components=2)
X_train_2dim = pd.DataFrame(pca.fit_transform(X_train),
                            columns = [['prin_comp1',
                                        'prin_comp2']])

X_test_2dim = pd.DataFrame(pca.transform(X_test),
                            columns = [['prin_comp1',
                                        'prin_comp2']])

X_train_1dim = X_train_2dim['prin_comp1']
X_test_1dim = X_test_2dim['prin_comp1']

X_train_2dim
X_train_1dim

# 2D plot of features

plt.scatter(X_train_2dim['prin_comp1'],X_train_2dim['prin_comp2'])
plt.title('2D plot of Training set after PCA')
plt.show()
# Combined 3D and 2D plot of features

fig = plt.figure()
ax = plt.axes(projection ='3d')

ax.scatter(X_train['variance'],X_train['skewness'],X_train['curtosis'],color='red')
ax.scatter(X_train_2dim['prin_comp1'],X_train_2dim['prin_comp2'],color='blue')
ax.set_title('Combined Plot')
plt.legend(labels=['Original 3D','After PCA 2D'])
plt.show()
# PCA Info

print('Info retained by PCA components (%) : ',
      [val*100 for val in pca.explained_variance_ratio_])
print()
print('Eigen Values : ',
      pca.explained_variance_)
print()
print('Eigen Vectors : ')
print(pca.components_)
# KNN On Original Dataset

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

print('Model performance on original training set : ')
print()
print(classification_report(y_train,
                            knn.predict(X_train)))

print()
print('Model performance on original test set : ')
print()
print(classification_report(y_test,
                            knn.predict(X_test)))
print()
print('Confusion Matrix : ')
cm_display = ConfusionMatrixDisplay.from_estimator(
             knn, X_test, y_test,
             display_labels=['0','1'])
plt.show()
# KNN On 2D PCA

knn = KNeighborsClassifier()
knn.fit(X_train_2dim,y_train)

print('Model performance after 2D PCA on training set : ')
print()
print(classification_report(y_train,
                            knn.predict(X_train_2dim)))

print()
print('Model performance after 2D PCA on test set : ')
print()
print(classification_report(y_test,
                            knn.predict(X_test_2dim)))
print()
print('Confusion Matrix : ')
cm_display = ConfusionMatrixDisplay.from_estimator(
             knn, X_test_2dim, y_test,
             display_labels=['0','1'])
plt.show()
# KNN On 1D PCA

knn = KNeighborsClassifier()
knn.fit(X_train_1dim,y_train)

print('Model performance after 1D PCA on training set : ')
print()
print(classification_report(y_train,
                            knn.predict(X_train_1dim)))

print()
print('Model performance after 1D PCA on test set : ')
print()
print(classification_report(y_test,
                            knn.predict(X_test_1dim)))
print()
print('Confusion Matrix : ')
cm_display = ConfusionMatrixDisplay.from_estimator(
             knn, X_test_1dim, y_test,
             display_labels=['0','1'])
plt.show()
result = pd.DataFrame({'Original':[100,100],
                      '2D PCA':[94,90],
                      '1D PCA':[78,75]},
                      index=['Training Accuracy (%)','Testing Accuracy (%)'])
print('Comparing Performance before and after PCA : \n')
result