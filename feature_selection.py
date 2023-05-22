
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from data_asset import data_frame, feature, createDataSetFile
import pandas as pd


print('\n\n===== Feature selection =====\n\n')


def Chi(x, y):
    k = 7
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(x, y)

    return X_new


def Mutual(x, y):
    k = 7
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(x, y)

    return X_new


def Recursive(x, y):
    k = 7
    estimator = LogisticRegression()
    rfe = RFE(estimator, n_features_to_select=k)
    X_new = rfe.fit_transform(x, y)

    return X_new


data = data_frame

xx = data.drop(['CMST_CASE_JUVENILE_REF', 'ครั้งที่กระทำความผิด'], axis=1)
yy = data['ครั้งที่กระทำความผิด']

# Perform feature selection using chi-square test
x_new_chi = Chi(xx, yy)
x_new_mutual = Mutual(xx, yy)
x_new_recursive = Recursive(xx, yy)

createDataSetFile(pd.DataFrame(x_new_chi),'chi.csv')
createDataSetFile(pd.DataFrame(x_new_mutual),'mutual.csv')
createDataSetFile(pd.DataFrame(x_new_recursive),'recursive.csv')

XX_features = [
  {
  'feature' : 'Chi square',
  'XX' : x_new_chi
  },
  {
  'feature' : 'Mutual information',
  'XX' : x_new_mutual
  },
  {
  'feature' : 'Recursive feature elimination',
  'XX' : x_new_recursive
  },
]

feature_selections = [
  {
  'feature_name' : 'Chi square',
  'feature_function' : Chi
  },
  {
  'feature_name' : 'Mutual information',
  'feature_function' : Mutual
  },
  {
  'feature_name' : 'Recursive feature elimination',
  'feature_function' : Recursive
  },
]