import pandas as pd
from sklearn.model_selection import train_test_split
# from joblib import dump

import data_asset
from model import knn_predict

print('\n\n===== k-Nearest Neighbors =====\n\n')

data = data_asset.data_frame

XX = pd.DataFrame(data, columns=data_asset.feature)
yy = pd.DataFrame(data['ครั้งที่กระทำความผิด'])


x_train, x_test, y_train, y_test = train_test_split(XX, yy, test_size=0.3, random_state=45)

knn_predict(x_train, x_test, y_train, y_test)
