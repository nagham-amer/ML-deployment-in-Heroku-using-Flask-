import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
tubing_p = pd.read_csv("tubing_547 +572 csv.csv")
T_P=tubing_p.dropna()
tubing_p=T_P.iloc[:,2:]
tubing_p=tubing_p.drop(['pco','pto','Ptv'],axis=1)
y = tubing_p['Pt@v']
X = tubing_p[['Pt@s','Tt@s','Pc@s', 'Tc@s','Tt@v','cycle time','injection time','oil density','depth','tubing diameter']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# reduce the number of the input data by using filtter by correlation method
def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname= corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features = correlation(X_train,0.8)
len(set(corr_features))
#drop the corr features from the X_test and X-train
X_train.drop(labels=corr_features,axis=1,inplace=True)
X_test.drop(labels=corr_features,axis=1,inplace=True)
X.drop(labels=corr_features,axis=1,inplace=True)

X_train.shape,X_test.shape
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
predictions_DT_train = tree_model.predict( X_train)
from sklearn.metrics import r2_score 
R = r2_score(y_train, predictions_DT_train)
print(R)
import pickle
pickle.dump(tree_model, open('model.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[1.4, 11, 9,126,60,15]]))