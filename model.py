import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# creating the model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import KBinsDiscretizer
import pickle

data = pd.read_csv("insurance.csv")

data.head()

data.info()

data.describe()

data.hist(bins=50, figsize=(20,15))
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(data['sex'])
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(data['smoker'])
plt.show()

plt.figure(figsize=(6,6))
sns.countplot(data['region'])
plt.show()

data.head()

data['sex'] = data['sex'].map({'female':0, 'male':1})
data['smoker'] = data['smoker'].map({'yes':1, 'no': 0})
data['region'] = data['region'].map({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4})

data.head()

features = data.select_dtypes(include=[np.number])
features.columns


# checking if any null values in  the data set
data.isnull().sum()

corrmat = data.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corrmat, annot=True, cmap="RdYlGn")

# so the data set is good lets train and test the data
x = data.drop(['charges'], axis=1)
y = data['charges']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

lr = LinearRegression()
lr.fit(x_train, y_train)
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

from sklearn import metrics
pred_lr = lr.predict(x_test)
ev_lr = metrics.r2_score(y_test, pred_lr)
pred_rf = rf.predict(x_test)
ev_rf = metrics.r2_score(y_test, pred_rf)

from sklearn import metrics
print("mean absolute error:", metrics.mean_absolute_error(y_test,pred_lr))
print("mean squared error:", metrics.mean_squared_error(y_test, pred_lr))
print("root mean squared error:", np.sqrt(metrics.mean_absolute_error(y_test, pred_lr)))

from sklearn import metrics
print("mean absolute error:", metrics.mean_absolute_error(y_test,pred_rf))
print("mean squared error:", metrics.mean_squared_error(y_test, pred_rf))
print("root mean squared error:", np.sqrt(metrics.mean_absolute_error(y_test, pred_rf)))

print("r2_score for lr", ev_lr)
print("r2_score for rf",ev_rf)

#cross validation
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(lr, x_train, y_train,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(rf, x_train, y_train,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

#fine-tunning my model
from sklearn.model_selection import GridSearchCV

param_grid = [
   
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

rf_1= RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_1, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(x_train, y_train)

grid_search.best_params_

grid_search.best_estimator_

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

rf_1.fit(x_train,y_train)
pred_rf_1 = rf_1.predict(x_test)

ev_rf_1 = metrics.r2_score(y_test, pred_rf_1)


print(ev_rf_1)

pickle.dump(rf_1, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))





































































