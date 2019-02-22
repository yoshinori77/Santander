import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report
import lightgbm as lgb
from sklearn.linear_model import ElasticNet, Lasso, Ridge, BayesianRidge, LassoLarsIC
from boruta import BorutaPy
import utils
import featuretools as ft
from sklearn.linear_model import LogisticRegression
from datetime import datetime
%matplotlib inline

pd.set_option("display.max_rows", 200)
pd.set_option('display.max_columns', 200)
pd.set_option('use_inf_as_na', True)

parent_dir = str(Path.cwd())

df = pd.read_csv(parent_dir + '/input/train.csv')
test_df = pd.read_csv(parent_dir + '/input/test.csv')

df.shape
test_df.shape
df.columns
df.select_dtypes(include='object')

sns.pairplot(df.iloc[:, :10], hue='target', plot_kws={'alpha': 0.2})
df.groupby('target').describe()

X = df.drop(['target', 'ID_code'], axis=1)
y = df.target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(objective='binary', num_leaves=10, class_weight='balanced',
                            learning_rate=0.05, n_estimators=5000,
                            boosting='gbdt', metric='auc')

model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
            early_stopping_rounds=100, verbose=100)
y_pred = model.predict(X_val)
roc_auc_score(y_val, y_pred)
accuracy_score(y_val, y_pred)
print(classification_report(y_val, y_pred))
confusion_matrix(y_val, y_pred)


log_dict = {'model': model, 'columns': X_train.columns,
            'roc_auc_score': roc_auc_score(y_val, y_pred),
            'accuracy_score': accuracy_score(y_val, y_pred),
            'classification_report': classification_report(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred)}
now = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

with open(parent_dir + '/output/log_' + now + '.pkl', mode='wb') as f:
    pickle.dump(log_dict, f)
