import pandas as pd
import os

from sklearn import preprocessing


le = preprocessing.LabelEncoder()

outdir = './temp_data'
original_file_name = 'original_dateset.csv'

colForConverstToInt = [
    'ฐานความผิดหลัก',
    'ฐานความผิดข้อหา',
    'ครั้งที่กระทำความผิด',  # target
    'อัตราโทษ',
    'กลุ่มASSIST',
    'SPECIAL_RN'
]

feature = [
    'ฐานความผิดหลัก',
    'ฐานความผิดข้อหา',
    'อายุขณะกระทำความผิด',
    'อัตราโทษ',
    'กลุ่มASSIST',
    'Risk',
    'Need',
    'SPECIAL_RN'

]

# data_frame.loc[:, colForConverstToInt]

# clean data
data_frame = pd.read_csv("../Data/clean/clean.csv", encoding="TIS-620")
for col in data_frame.loc[:, colForConverstToInt]:
    data_frame[col] = le.fit_transform(data_frame[col])

# clean and balance data
data_frame1 = pd.read_csv(
    "../Data/clean/split_under_data/under_01.csv", encoding="TIS-620")
for col in data_frame1.loc[:, colForConverstToInt]:
    data_frame1[col] = le.fit_transform(data_frame1[col])

data_frame2 = pd.read_csv(
    "../Data/clean/split_under_data/under_02.csv", encoding="TIS-620")
for col in data_frame2.loc[:, colForConverstToInt]:
    data_frame2[col] = le.fit_transform(data_frame2[col])

data_frame3 = pd.read_csv(
    "../Data/clean/split_under_data/under_03.csv", encoding="TIS-620")
for col in data_frame3.loc[:, colForConverstToInt]:
    data_frame3[col] = le.fit_transform(data_frame3[col])

data_frame4 = pd.read_csv(
    "../Data/clean/split_under_data/under_04.csv", encoding="TIS-620")
for col in data_frame4.loc[:, colForConverstToInt]:
    data_frame4[col] = le.fit_transform(data_frame4[col])

data_frame5 = pd.read_csv(
    "../Data/clean/split_under_data/under_05.csv", encoding="TIS-620")
for col in data_frame5.loc[:, colForConverstToInt]:
    data_frame5[col] = le.fit_transform(data_frame5[col])

#column = CMST_CASE_JUVENILE_REF, ฐานความผิดหลัก, ฐานความผิดข้อหา, ครั้งที่กระทำความผิด, อายุขณะกระทำความผิด, อัตราโทษ, กลุ่มASSIST, Risk, Need, SPECIAL_RN


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


def createDataSetFile(data,file_name):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    data.loc[:, ].to_csv(
        '{}/{}'.format(outdir, file_name), index=False)


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

