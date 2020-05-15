

# # Importing The Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing the Dataset

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])

sns.countplot(df_cancer['target'])

sns.scatterplot(x = 'mean perimeter', y = 'mean texture', hue = 'target', data = df_cancer)

plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(), annot = True)

x = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 5)

min_train = x_train.min()
range_train = (x_train - min_train).max()
x_train_sc = (x_train - min_train)/range_train


sns.scatterplot(x = x_train['mean perimeter'], y = x_train['mean texture'], hue = y_train)
sns.scatterplot(x = x_train_sc['mean perimeter'], y = x_train_sc['mean texture'], hue = y_train)

min_test = x_test.min()
range_test = (x_test - min_test).max()
x_test_sc = (x_test - min_test)/range_test


from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train_sc, y_train)


# # Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report
y_pred = classifier.predict(x_test_sc)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, y_pred))

# # Improving the model
param_grid = {'C' : [0.1, 1, 10, 100], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel' : ['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)
grid.fit(x_train_sc, y_train)
grid.best_params_
grid_pred = grid.predict(x_test_sc)
cm = confusion_matrix(y_test, grid_pred)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, grid_pred))

