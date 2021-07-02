import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('../data/training_set/xgb_reg_data.csv')

data = dataset.drop(['Unnamed: 0'],axis=1)






X = data.drop(['arr_delay'], axis=1)
y = data['arr_delay']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

pca = PCA(n_components=4) # estimate only 4 PCs
X_new = pca.fit_transform(X) # project the original data into the 
X_pca = pd.DataFrame(X_new)

X_pca.to_csv('../data/training_set/xgb_reg_pca.csv')
