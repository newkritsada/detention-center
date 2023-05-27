
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from data_asset import data_frame,DataExcept1, DataExcept2, DataExcept3, DataExcept4, DataExcept5
from data_asset_generate import create_data_file,outdir,split_data
import pandas as pd


# print('\n\n===== Feature selection =====\n\n')


def Chi(x, y):
    k = 7
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(x, y)

    selected_cols_indices = selector.get_support(indices=True)
    selected_cols = x.columns[selected_cols_indices]
    X_selected = pd.DataFrame(X_new, columns=selected_cols)

    return X_selected


def Mutual(x, y):
    k = 7
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(x, y)

    selected_cols_indices = selector.get_support(indices=True)
    selected_cols = x.columns[selected_cols_indices]
    X_selected = pd.DataFrame(X_new, columns=selected_cols)

    return X_selected


def Recursive(x, y):
    k = 7
    estimator = LogisticRegression()
    rfe = RFE(estimator, n_features_to_select=k)
    X_new = rfe.fit_transform(x, y)

    selected_cols_indices = rfe.get_support(indices=True)
    selected_cols = x.columns[selected_cols_indices]
    X_selected = pd.DataFrame(X_new, columns=selected_cols)

    return X_selected

def getXY(data):
    xx = data.drop(['CMST_CASE_JUVENILE_REF', 'ครั้งที่กระทำความผิด'], axis=1)
    yy = data['ครั้งที่กระทำความผิด']
    return xx,yy

data = data_frame

xx,yy = getXY(data)

# Perform feature selection using chi-square test
x_new_chi = Chi(xx, yy)
x_new_mutual = Mutual(xx, yy)
x_new_recursive = Recursive(xx, yy)

path = outdir+'/feature_selection'
split = '/split_data'

create_data_file(path,'chi-7.csv',pd.DataFrame(x_new_chi))
create_data_file(path,'mutual-7.csv',pd.DataFrame(x_new_mutual))
create_data_file(path,'recursive-7.csv',pd.DataFrame(x_new_recursive))


data_split = [DataExcept1(), DataExcept2(), DataExcept3(),
               DataExcept4(), DataExcept5()]
feature_array = ['chi','mutual','recursive']
feature_array_func = [Chi,Mutual,Recursive]

for i in range(len(feature_array)):
  for index,data_item in enumerate(data_split):
      xx,yy = getXY(data_item)
      x_new = feature_array_func[i](xx,yy)
      create_data_file(path+split, '{}-{}.csv'.format(feature_array[i],index+1),pd.DataFrame(x_new))
