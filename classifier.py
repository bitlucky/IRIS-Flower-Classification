import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split
x_train , y_train , x_test , y_test = train_test_split(x, y , test_size = 0.3 , random_state=0)

from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators = 10 , criterion='entropy' , random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)

from sklearn.metrics import accuracy_score as acs
print(acs(y_pred, y_test))