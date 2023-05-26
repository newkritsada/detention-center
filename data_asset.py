import pandas as pd
import os

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

outdir = './temp_data'
original_file_name = 'original_dateset.csv'

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

# Feature for smote
# features = [
#     "maincrime",
#     "crime",
#     "age",
#     "result",
#     "assist",
#     "Risk",
#     "Need",
# ]

data_frame_convert = [
    'ฐานความผิดหลัก',
    'ฐานความผิดข้อหา',
    'ครั้งที่กระทำความผิด',  # target
    'อัตราโทษ',
    'กลุ่มASSIST',
    'SPECIAL_RN',
]

# data_frame_smote_convert = [
#     'SPECIAL_RN',
# ]

data_frame_split_convert = [
    'SPECIAL_RN',
]



# clean data
data_frame = pd.read_csv("../Data/clean/clean.csv", encoding="TIS-620")
# data_frame = pd.read_csv("../Data/clean/smote/smote_data.csv", encoding="TIS-620")
for col in data_frame.loc[:, data_frame_convert]:
    data_frame[col] = le.fit_transform(data_frame[col])

# clean and balance data
data_frame1 = pd.read_csv(
    "../Data/clean/split_under_data/under_01.csv", encoding="TIS-620")
# data_frame1 = pd.read_csv(
#     "../Data/clean/smote/split_data/smote01.csv", encoding="TIS-620")
for col in data_frame1.loc[:, data_frame_split_convert]:
    data_frame1[col] = le.fit_transform(data_frame1[col])

data_frame2 = pd.read_csv(
    "../Data/clean/split_under_data/under_02.csv", encoding="TIS-620")
# data_frame2 = pd.read_csv(
#     "../Data/clean/smote/split_data/smote02.csv", encoding="TIS-620")
for col in data_frame2.loc[:, data_frame_split_convert]:
    data_frame2[col] = le.fit_transform(data_frame2[col])

data_frame3 = pd.read_csv(
    "../Data/clean/split_under_data/under_03.csv", encoding="TIS-620")
# data_frame3 = pd.read_csv(
#     "../Data/clean/smote/split_data/smote03.csv", encoding="TIS-620")
for col in data_frame3.loc[:, data_frame_split_convert]:
    data_frame3[col] = le.fit_transform(data_frame3[col])

data_frame4 = pd.read_csv(
    "../Data/clean/split_under_data/under_04.csv", encoding="TIS-620")
# data_frame4 = pd.read_csv(
#     "../Data/clean/smote/split_data/smote04.csv", encoding="TIS-620")
for col in data_frame4.loc[:, data_frame_split_convert]:
    data_frame4[col] = le.fit_transform(data_frame4[col])

data_frame5 = pd.read_csv(
    "../Data/clean/split_under_data/under_05.csv", encoding="TIS-620")
# data_frame5 = pd.read_csv(
#     "../Data/clean/smote/split_data/smote05.csv", encoding="TIS-620")
for col in data_frame5.loc[:, data_frame_split_convert]:
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



createDataSetFile(data_frame5,original_file_name)