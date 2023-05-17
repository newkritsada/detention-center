import pandas as pd
import os
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


outdir = './cleanData'
le = preprocessing.LabelEncoder()

colForConverstToInt = [
    'ฐานความผิดหลัก',
    'ฐานความผิดข้อหา',
    'ครั้งที่กระทำความผิด',  # target
    'อัตราโทษ',
    'กลุ่มASSIST'
]
# data_frame.loc[:, colForConverstToInt]

# clean data
data_frame = pd.read_csv("../Data/clean/clean.csv", encoding="TIS-620")
for col in data_frame.loc[:, colForConverstToInt]:
    data_frame[col] = le.fit_transform(data_frame[col])

# clean and balance data
data_frame1 = pd.read_csv(
    "../Data/clean/split_data/data1.csv", encoding="TIS-620")
for col in data_frame1.loc[:, colForConverstToInt]:
    data_frame1[col] = le.fit_transform(data_frame1[col])

data_frame2 = pd.read_csv(
    "../Data/clean/split_data/data2.csv", encoding="TIS-620")
for col in data_frame2.loc[:, colForConverstToInt]:
    data_frame2[col] = le.fit_transform(data_frame2[col])

data_frame3 = pd.read_csv(
    "../Data/clean/split_data/data3.csv", encoding="TIS-620")
for col in data_frame3.loc[:, colForConverstToInt]:
    data_frame3[col] = le.fit_transform(data_frame3[col])

data_frame4 = pd.read_csv(
    "../Data/clean/split_data/data4.csv", encoding="TIS-620")
for col in data_frame4.loc[:, colForConverstToInt]:
    data_frame4[col] = le.fit_transform(data_frame4[col])

data_frame5 = pd.read_csv(
    "../Data/clean/split_data/data5.csv", encoding="TIS-620")
for col in data_frame5.loc[:, colForConverstToInt]:
    data_frame5[col] = le.fit_transform(data_frame5[col])

#column = CMST_CASE_JUVENILE_REF, ฐานความผิดหลัก, ฐานความผิดข้อหา, ครั้งที่กระทำความผิด, อายุขณะกระทำความผิด, อัตราโทษ, กลุ่มASSIST, Risk, Need, SPECIAL_RN

feature = [
    # 'CMST_CASE_JUVENILE_REF',
    'ฐานความผิดหลัก',
    'ฐานความผิดข้อหา',
    'อายุขณะกระทำความผิด',
    'อัตราโทษ',
    'กลุ่มASSIST',
    'Risk',
    'Need',
]


def DataExcept1():
    return pd.concat([data_frame2, data_frame3, data_frame4, data_frame5])


def DataExcept2():
    return pd.concat([data_frame1, data_frame3, data_frame4, data_frame5])


def DataExcept3():
    return pd.concat([data_frame1, data_frame2, data_frame4, data_frame5])


def DataExcept4():
    return pd.concat([data_frame1, data_frame2, data_frame3, data_frame5])


def DataExcept5():
    return pd.concat([data_frame1, data_frame2, data_frame3, data_frame4])


def createOriginalDataSet():
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    data_frame.loc[:, ].to_csv(
        '{}/{}'.format(outdir, 'original_dateset.csv'), index=False)


def formatDataset(colForConverstToInt):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # filter only column have to use

    # convert string to int
    columns = data_frame.loc[:, colForConverstToInt]

    for col in columns:
        columns.loc[:, col], mapping = pd.factorize(columns[col])

    # create temp dataset for reverse after test data
    new_data_frame = data_frame.copy()
    new_data_frame.loc[:, colForConverstToInt] = columns.loc[:,
                                                             colForConverstToInt]

    # create file for new data

    new_data_frame.loc[:, ].to_csv(
        '{}/{}'.format(outdir, 'new_dataset.csv'), index=False)

    for col in columns:
        values = pd.DataFrame(
            {'old_value': data_frame[col], 'new_value': columns[col]})
        values = values.drop_duplicates(subset=['old_value', 'new_value'])
        file_name = '{}_{}_rows_values.csv'.format(col, len(data_frame))
        values.to_csv('{}/{}'.format(outdir, file_name), index=False)

    print("\nChange value to number with auto map success!\n\n")
    return new_data_frame

# Feature selection


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