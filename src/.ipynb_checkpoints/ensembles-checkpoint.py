
import pandas as pd
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)
import datetime, warnings, scipy
warnings.filterwarnings("ignore")

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error as mae
from sklearn import preprocessing

import imblearn
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

# import packages for hyperparameters tuning
#from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

dataset = pd.read_csv('../data/training_set/TRAIN_ME_carriers.csv')
data = dataset.copy()
data['mkt_carrier'].unique()


le = preprocessing.LabelEncoder()
data['mkt_carrier'] = le.fit_transform(data['mkt_carrier'])

data = data.drop(['Unnamed: 0'],axis=1)
data['flight_status'] = (data['arr_delay'] > 15).replace([True,False],[1,0])
data.drop(['arr_delay'],axis=1,inplace=True)
data['speed']=data['distance']/data['crs_elapsed_time']


sample = data.sample(n=200000)


X = sample.drop(['flight_status'], axis=1)
y = sample['flight_status']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#### SMOTE sampling

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

#### XGBoost Basic
# clf = xgb.XGBClassifier()
# clf.fit(X_train,y_train)

# y_pred = clf.predict(X_test)


# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# acc = accuracy_score(y_test, y_pred)
# print(acc)



# Fitting the model and calculating the training and text (val) accuracies
# #### Imbalanced Classification
# clf = xgb.XGBClassifier()


# parameters = {'learning_rate': [0.1],
#                   'max_depth': [6,10],
#                   'min_child_weight': [4,7],
#                   'scale_pos_weight': [.1,.4,.75],
#                   'alpha': [0.01,.05],
#                   'objective': ['binary:logistic'],
#                   'subsample': [0.7], 
#                   'n_estimators': [100,1000]
              
              
#              }



# grid_clf = GridSearchCV(clf, parameters, scoring='accuracy', cv=None, n_jobs=1)
# grid_clf.fit(X_train_res, y_train_res)

# best_parameters = grid_clf.best_params_

# print("Grid Search found the following optimal parameters: ")
# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))

# training_preds = grid_clf.predict(X_train_res)
# val_preds = grid_clf.predict(X_test)
# training_accuracy = accuracy_score(y_train_res, training_preds)
# val_accuracy = accuracy_score(y_test, val_preds)

# print("")
# print("Training Accuracy: {:.4}%".format(training_accuracy * 100))
# print("Validation accuracy: {:.4}%".format(val_accuracy * 100))

# #### XGBoost
model_imb = xgb.XGBClassifier(alpha=.01,
                             learning_rate = 0.1,
                             max_depth = 10,
                             min_child_weight = 7,
                             n_estimators = 100,
                             objective = 'binary:logistic',
                             scale_pos_weight = 0.75,
                             subsample = 0.7)
                                   
model_imb.fit(X_train_res, y_train_res)

print("Accuracy on training set: {:.2f}".format(model_imb.score(X_train, y_train) * 100))
print("Accuracy on validation set: {:.2f}".format(model_imb.score(X_test, y_test) * 100))

xgb_predict = model_imb.predict(X_test)
xgb_predict

print(confusion_matrix(y_test,xgb_predict))
print(classification_report(y_test,xgb_predict))


xgb.plot_importance(model_imb)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

# save in JSON format
model_imb.save_model("xgb_classifier.json")
"""
Training the whole dataset:
Accuracy on training set: 81.07
Accuracy on validation set: 80.91
[[619955   8124]
 [140753  10836]]
              precision    recall  f1-score   support

           0       0.81      0.99      0.89    628079
           1       0.57      0.07      0.13    151589

    accuracy                           0.81    779668
   macro avg       0.69      0.53      0.51    779668
weighted avg       0.77      0.81      0.74    779668

"""




#######################################################################

"""

alpha: 0.01
eval_metric: ‘mae’
learning_rate: 0.1
max_depth: 6
min_child_weight: 7
n_estimators: 100
objective: ‘binary:logistic’
scale_pos_weight: 0.9
subsample: 0.7


Accuracy on training set: 80.63
Accuracy on validation set: 80.55
[[321783    217]
 [ 77603    397]]
              precision    recall  f1-score   support

           0       0.81      1.00      0.89    322000
           1       0.65      0.01      0.01     78000

    accuracy                           0.81    400000
   macro avg       0.73      0.50      0.45    400000
weighted avg       0.77      0.81      0.72    400000



alpha: 0.01
learning_rate: 0.1
max_depth: 6
min_child_weight: 9
n_estimators: 100
objective: 'binary:logistic'
scale_pos_weight: 0.75
subsample: 0.7


Accuracy on training set: 80.61
Accuracy on validation set: 80.52
[[321933     67]
 [ 77856    144]]
              precision    recall  f1-score   support

           0       0.81      1.00      0.89    322000
           1       0.68      0.00      0.00     78000

    accuracy                           0.81    400000
   macro avg       0.74      0.50      0.45    400000
weighted avg       0.78      0.81      0.72    400000





Accuracy on training set: 80.60
Accuracy on validation set: 80.52
[[321958     42]
 [ 77884    116]]
              precision    recall  f1-score   support

           0       0.81      1.00      0.89    322000
           1       0.73      0.00      0.00     78000

    accuracy                           0.81    400000
   macro avg       0.77      0.50      0.45    400000
weighted avg       0.79      0.81      0.72    400000



alpha: 0.01
learning_rate: 0.1
max_depth: 6
min_child_weight: 7
n_estimators: 100
objective: 'binary:logistic'
scale_pos_weight: 0.75
subsample: 0.7

Training Accuracy: 87.65%
Validation accuracy: 80.8%

ccuracy on training set: 81.23
Accuracy on validation set: 80.75
[[3225   14]
 [ 756    5]]
              precision    recall  f1-score   support

           0       0.81      1.00      0.89      3239
           1       0.26      0.01      0.01       761

    accuracy                           0.81      4000
   macro avg       0.54      0.50      0.45      4000
weighted avg       0.71      0.81      0.73      4000


Grid Search found the following optimal parameters: 
alpha: 0.01
learning_rate: 0.1
max_depth: 10
min_child_weight: 7
n_estimators: 100
objective: 'binary:logistic'
scale_pos_weight: 0.75
subsample: 0.7

Training Accuracy: 89.94%
Validation accuracy: 80.55%

Accuracy on training set: 81.24
Accuracy on validation set: 80.80
[[3225   14]
 [ 754    7]]
              precision    recall  f1-score   support

           0       0.81      1.00      0.89      3239
           1       0.33      0.01      0.02       761

    accuracy                           0.81      4000
   macro avg       0.57      0.50      0.46      4000
weighted avg       0.72      0.81      0.73      4000
"""