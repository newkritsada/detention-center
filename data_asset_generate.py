import pandas as pd
import os

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

outdir = './temp_data'

def encode_data(data,col_convert):
    for col in data.loc[:, col_convert]:
        if col=='ครั้งที่กระทำความผิด' and data[col][0]=='ครั้งแรก':
            data[col] = le.fit_transform(data[col])
            data[col] = [1 if value == 0 else 0 for value in data[col]]
        else:
            data[col] = le.fit_transform(data[col])

    return data

def revers_data(data):
    data['ครั้งที่กระทำความผิด'] = data['ครั้งที่กระทำความผิด'].apply(lambda value: 1 if value == 0 else 0)
    return data

def create_data_file(path,file_name,data):
    if not os.path.exists(path):
        os.mkdir(path)
    data.loc[:, ].to_csv(
        '{}/{}'.format(path, file_name), index=False)

data_frame_convert = [
    'ฐานความผิดหลัก',
    'ฐานความผิดข้อหา',
    'ครั้งที่กระทำความผิด',  # target
    'อัตราโทษ',
    'กลุ่มASSIST',
    'SPECIAL_RN',
]

data_frame_split_convert = [
    'SPECIAL_RN',
]

# data_frame_all_smote_convert = [
#     'SPECIAL_RN',
# ]


data_path = "../Data/clean/"
# data_path = "../Data/clean/smote/"
data_file = "clean.csv"
# data_file = "smote_data.csv"
# clean data
data_frame = pd.read_csv(data_path+data_file, encoding="TIS-620")
data_frame = encode_data(data_frame,data_frame_convert)
create_data_file(outdir ,data_file,data_frame)
#==================================

# clean and balance data
data_path_split = "../Data/clean/split_under_data/"
# data_path_split = "../Data/clean/smote/split_data/"
split_data = '/split_data'
# split_data = '/smote/split_data'


data_file_1 = "under_01.csv"
data_file_2 = "under_02.csv"
data_file_3 = "under_03.csv"
data_file_4 = "under_04.csv"
data_file_5 = "under_05.csv"

##### Smote
# data_file_1 = "SMOTE1.csv"
# data_file_2 = "SMOTE2.csv"
# data_file_3 = "SMOTE3.csv"
# data_file_4 = "SMOTE4.csv"
# data_file_5 = "SMOTE5.csv"

data_frame1 = pd.read_csv(data_path_split + data_file_1, encoding="TIS-620")
data_frame1 = encode_data(data_frame1,data_frame_split_convert)
data_frame1 = revers_data(data_frame1)
create_data_file(outdir + split_data,data_file_1,data_frame1)

#==================================

data_frame2 = pd.read_csv(data_path_split + data_file_2, encoding="TIS-620")
data_frame2 = encode_data(data_frame2,data_frame_split_convert)
data_frame2 = revers_data(data_frame2)
create_data_file(outdir + split_data,data_file_2,data_frame2)

#==================================

data_frame3 = pd.read_csv(data_path_split + data_file_3, encoding="TIS-620")
data_frame3 = encode_data(data_frame3,data_frame_split_convert)
data_frame3 = revers_data(data_frame3)
create_data_file(outdir + split_data,data_file_3,data_frame3)

#==================================

data_frame4 = pd.read_csv(data_path_split + data_file_4, encoding="TIS-620")
data_frame4 = encode_data(data_frame4,data_frame_split_convert)
data_frame4 = revers_data(data_frame4)
create_data_file(outdir + split_data,data_file_4,data_frame4)

#==================================

data_frame5 = pd.read_csv(data_path_split + data_file_5, encoding="TIS-620")
data_frame5 = encode_data(data_frame5,data_frame_split_convert)
data_frame5 = revers_data(data_frame5)
create_data_file(outdir + split_data,data_file_5,data_frame5)

