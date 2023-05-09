import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

data = pd.read_csv("../Data/clean/clean.csv", encoding="TIS-620")
le = preprocessing.LabelEncoder()
for col in data:
    data[col] = le.fit_transform(data[col])
#แยกminor/major
minority_data = data[data['ครั้งที่กระทำความผิด'] == 'กระทำผิดซ้ำ']
majority_data = data[data['ครั้งที่กระทำความผิด'] == 'ครั้งแรก']

# featureslist
features = ['ฐานความผิดหลัก', 'ฐานความผิดข้อหา', 'อายุขณะกระทำความผิด', 'อัตราโทษ', 'กลุ่มASSIST', 'Risk' ,  'Need']

# Separate the features from the target variable for both minority and majority classes
minority_features = minority_data[features]
minority_target = minority_data['ครั้งที่กระทำความผิด']
majority_features = majority_data[features]
majority_target = majority_data['ครั้งที่กระทำความผิด']

smote = SMOTE()
oversampled_minority_features, oversampled_minority_target = smote.fit_resample(minority_features, minority_target)


balanced_features = pd.concat([oversampled_minority_features, majority_features])
balanced_target = pd.concat([oversampled_minority_target, majority_target])

# รวมDataFrameใหม่
balanced_data = pd.concat([balanced_features, balanced_target], axis=1)


balanced_data.to_csv('balanced_data.csv', index=False)