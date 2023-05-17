import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


import data_asset
from model import dicision_tree_predict

print('\n\n===== Dicision Tree =====\n\n')

data = data_asset.data_frame

le = preprocessing.LabelEncoder()
for col in data.loc[1:]:
    data[col] = le.fit_transform(data[col])

XX = pd.DataFrame(data, columns=data_asset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])


x_train, x_test, y_train, y_test = train_test_split(
    XX, yy, test_size=0.3, random_state=45)

dicision_tree_predict(x_train, x_test, y_train, y_test)
