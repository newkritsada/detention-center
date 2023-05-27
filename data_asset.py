import pandas as pd

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

data_path = "./temp_data/"
# data_path = "../Data/clean/smote/"
data_file = "clean.csv"
# data_file = "smote_data.csv"

data_frame = pd.read_csv(data_path+data_file, encoding="utf-8")
#==================================

# clean and balance data
data_path_split = "./temp_data/split_data/"

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

data_frame1 = pd.read_csv(data_path_split + data_file_1, encoding="utf-8") # "TIS-620"
#==================================
data_frame2 = pd.read_csv(data_path_split + data_file_2, encoding="utf-8")
#==================================
data_frame3 = pd.read_csv(data_path_split + data_file_3, encoding="utf-8")
#==================================
data_frame4 = pd.read_csv(data_path_split + data_file_4, encoding="utf-8")
#==================================
data_frame5 = pd.read_csv(data_path_split + data_file_5, encoding="utf-8")


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