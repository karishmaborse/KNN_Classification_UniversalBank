# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('UniversalBank.csv')
X = dataset.iloc[:, [1,2,3,5,6,7,8,9,10,11,12]].values
y = dataset.iloc[:, 13].values
#Categorical feature scaling, Education needs to be scaled.
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X = LabelEncoder()
X[:,5] = labelencoder_X.fit_transform(X[:,5])
onehotencoder_X = OneHotEncoder(categorical_features=[3])
X=onehotencoder_X.fit_transform(X).toarray()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 20, p=2, metric = 'minkowski')
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#calculate the accuracy 
from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
#Accuracy = (TP+TN)/(Total elements)
Accuracy = ((1798+121)/2000)
#Question B- to check the accuracy of the k value
# Fitting classifier to the Training set with K= 1
from sklearn.neighbors import KNeighborsClassifier
classifier_1 = KNeighborsClassifier(n_neighbors = 20, p=2, metric = 'minkowski')
classifier_1.fit(X_train,y_train)
#Question A
# Importing the dataset
dataset_Scenario1 = pd.read_csv('scenario1.csv')
Scenario1_X = dataset_Scenario1.iloc[:, :14].values
'''#Fitting the data into the scales
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Scenario1_X_train = sc.fit_transform(Scenario1_X)'''

Y_Scenario_pred = classifier_1.predict(Scenario1_X)

#question e Spliting the data sets into  Train , Test , Validate.
from sklearn.model_selection import train_test_split

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.2, random_state=1)

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(X_train_new, y_train_new, test_size=0.3, random_state=1)

#Applying KNN for the new datasets.
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_new = sc.fit_transform(X_train_new)
X_val_new = sc.transform(X_val_new)
# Fitting classifier to the Training set K=3 (having highest accuracy )
from sklearn.neighbors import KNeighborsClassifier
classifier_new = KNeighborsClassifier(n_neighbors = 3, p=2, metric = 'minkowski')
classifier_new.fit(X_train_new,y_train_new)
#Predicting the values using new Validation dataset.
y_pred_new = classifier_new.predict(X_val_new)
#Calculating the new Confusion matrix.
from sklearn.metrics import confusion_matrix
cm_new = confusion_matrix( y_val_new, y_pred_new)
#calculate the accuracy 
from sklearn.metrics import accuracy_score
accuracy_score_new = accuracy_score(y_val_new, y_pred_new, normalize=True, sample_weight=None)
