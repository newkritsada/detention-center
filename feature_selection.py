
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression

import pandas as pd
from data_asset import data_frame1, data_frame2, data_frame3, data_frame4, data_frame5


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


path = './temp_data/feature_selection/'
chi = 'chi-7.csv'
mutual = 'mutual-7.csv'
recursive = 'recursive-7.csv'


feature_array = ['chi', 'mutual', 'recursive']

XX_features = [
    {
        'feature': 'Chi square',
        'XX':  pd.read_csv(path+chi, encoding="utf-8")
    },
    {
        'feature': 'Mutual information',
        'XX': pd.read_csv(path+mutual, encoding="utf-8")
    },
    {
        'feature': 'Recursive feature elimination',
        'XX': pd.read_csv(path+recursive, encoding="utf-8")
    },
]


feature_selection_names = [
    'Chi square', 'Mutual information', 'Recursive feature elimination']

data_path = './temp_data/feature_selection/split_data/'


def get_file(path, name):
    return pd.read_csv(path + name, encoding="utf-8")


def data_except_select(items, index):
    return pd.concat([item for i, item in enumerate(items) if i != index])


data_frame_chi_train = [
    get_file(data_path, 'chi-1.csv'),
    get_file(data_path, 'chi-2.csv'),
    get_file(data_path, 'chi-3.csv'),
    get_file(data_path, 'chi-4.csv'),
    get_file(data_path, 'chi-5.csv'),
]

data_frame_mutual_train = [
    get_file(data_path, 'mutual-1.csv'),
    get_file(data_path, 'mutual-2.csv'),
    get_file(data_path, 'mutual-3.csv'),
    get_file(data_path, 'mutual-4.csv'),
    get_file(data_path, 'mutual-5.csv'),
]

data_frame_recursive_train = [
    get_file(data_path, 'recursive-1.csv'),
    get_file(data_path, 'recursive-2.csv'),
    get_file(data_path, 'recursive-3.csv'),
    get_file(data_path, 'recursive-4.csv'),
    get_file(data_path, 'recursive-5.csv'),
]

data_tests = [data_frame1, data_frame2, data_frame3, data_frame4, data_frame5]

feature_selections = [
    {
        'feature_name': 'Chi square',
        'data_trains': data_frame_chi_train,
        'data_tests': data_tests
    },
    {
        'feature_name': 'Mutual information',
        'data_trains': data_frame_mutual_train,
        'data_tests': data_tests
    },
    {
        'feature_name': 'Recursive feature elimination',
        'data_trains': data_frame_recursive_train,
        'data_tests': data_tests
    },
]
