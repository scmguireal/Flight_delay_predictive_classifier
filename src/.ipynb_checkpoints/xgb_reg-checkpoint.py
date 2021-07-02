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
from sklearn.decomposition import PCA
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

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


dataset = pd.read_csv('../data/training_set/xgb_reg.csv')
data = dataset.copy()
# data['mkt_carrier'].unique()



# data = data.drop(['Unnamed: 0'],axis=1)
# data['speed']=data['distance']/data['crs_elapsed_time']

# data = pd.get_dummies(data, columns=['mkt_carrier'], prefix='mkt_carrier', drop_first=True)
# data = pd.get_dummies(data, columns=['dest_airport_id'], prefix='DEST', drop_first=True)
# data = pd.get_dummies(data, columns=['weather_desc'], prefix='weather', drop_first=True) 
# data = pd.get_dummies(data, columns=['month'], prefix='month', drop_first=True)
# data = pd.get_dummies(data, columns=['day'], prefix='day', drop_first=True)


# data.to_csv('../data/training_set/xgb_reg.csv')

sample = data.sample(n=100000)


X = sample.drop(['arr_delay'], axis=1)
y = sample['arr_delay']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# The PCA model
pca = PCA(n_components=4) # estimate only 4 PCs
X_new = pca.fit_transform(X) # project the original data into the 
X_pca = pd.DataFrame(X_new)

X_pca.to_csv('../data/training_set/xgb_reg_pca.csv')

# data_dmatrix = xgb.DMatrix(data=X,label=y)

# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size = 0.2, random_state = 0)


xg_reg = xgb.XGBRegressor(objective ='reg:linear', 
                          colsample_bytree = 0.3, 
                          learning_rate = 0.1,
                          max_depth = 5, 
                          alpha = 10, 
                          n_estimators = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test,preds
             )
print("RMSE: %f" % (rmse))
print("r2 score: %f" % (r2))


#############################
# params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
#                 'max_depth': 5, 'alpha': 10}

# cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
#                     num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

# cv_results.head()

# print((cv_results["test-rmse-mean"]).tail(1))

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

# model = XGBRegressor()
# # define model evaluation method
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# # evaluate model
# scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# # force scores to be positive
# scores = absolute(scores)
# print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )