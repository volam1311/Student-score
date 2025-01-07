import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data = pd.read_csv("StudentScore.xls")
target = "math score"
#print(data.info())
#print(data.corr())
"""
sns.histplot(data["math score"])
plt.title("Math score distribution")
plt.savefig("ss1.jpg")
"""
x = data.drop(target,axis = 1)
y = data[target]
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state = 42)
#print(x["gender"].unique())
#print(x["race/ethnicity"].unique())
"""
imputer = SimpleImputer(strategy = 'mean')
x["reading score"] = imputer.fit_transform(x[["reading score"]])
print(x["reading score"])
"""
#scaler = StandardScaler()
num_transformer = Pipeline(steps = [
    ("imputer",SimpleImputer(strategy = "median")),
    ("scaler",StandardScaler())
])
"""
result = num_transformer.fit_transform(x_train[["reading score"]])
for i,j in zip(x_train["reading score"],result):
    print("Before {} After {}".format(i,j))
"""
gender_values = ["male","female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
education_values = ["some high school","high school","some college","associate's degree","bachelor's degree","master's degree"]
ordinal_transformer = Pipeline(steps = [
    ("imputer",SimpleImputer(strategy = "most_frequent",fill_value = "unknown")),
    ("scaler",OrdinalEncoder(categories= [education_values,gender_values,lunch_values,test_values]))
])
"""
result = ordinal_transformer.fit_transform(x_train[["parental level of education"]])
for i,j in zip(x_train["parental level of education"],result):
    print("Before {} After {}".format(i,j))
"""
nominal_transformer = Pipeline(steps = [
    ("imputer",SimpleImputer(strategy = "most_frequent",fill_value = "unknown")),
    ("encoder",OneHotEncoder( sparse_output =   False ))
])
"""
result = nominal_transformer.fit_transform(x_train[["race/ethnicity"]])
for i,j in zip(x_train["race/ethnicity"],result):
    print("Before {} After {}".format(i,j))
"""
preprocessor = ColumnTransformer(transformers = [
    ("num_features",num_transformer,["reading score","writing score"]),
    ("ordinal_features",ordinal_transformer,["parental level of education","gender","lunch","test preparation course"]),
    ("nominal_features",nominal_transformer,["race/ethnicity"]),
])
rgs = Pipeline(steps = [
    ("preprocessor",preprocessor),
    ("regressor",RandomForestRegressor())
])
"""
rgs.fit(x_train,y_train)
y_predict = rgs.predict(x_test)
"""
#for i,j in zip(y_test,y_predict):
#    print("Predict {} Actual {}".format(i,j))
"""
print("MAE {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE {}".format(mean_squared_error(y_test, y_predict)))
print("r2_score {}".format(r2_score(y_test, y_predict)))
"""

parameters= {
    "regressor__n_estimators" : [50,100,200],
    "regressor__criterion" :["squared_error","absolute_error"],
    "regressor__max_depth" :[None,5,10],
    "regressor__max_features": ["sqrt","log2"],
}
#model = GridSearchCV(rgs,param_grid = parameters,scoring = "r2", cv= 6, verbose = 1,n_jobs = 6)
model = RandomizedSearchCV(rgs,param_distributions= parameters,scoring = "r2", cv= 6, verbose = 1,n_jobs = 6,n_iter=20)
model.fit(x_train,y_train)
print(model.best_score_)
print(model.best_params_)

"""
lazy_rgs = LazyRegressor(verbose =0, ignore_warnings= False,custom_metric= None)
models,predictions = lazy_rgs.fit(x_train, x_test, y_train, y_test)
print(predictions)
"""