import pandas as pd
from imblearn.over_sampling import SMOTE

# อ่านข้อมูลจากไฟล์ CSV
data = pd.read_csv('original_dataset.csv')

# แยกข้อมูลเชิงปริมาณและเป้าหมาย
X = data.drop(['repeated'], axis=1)
y = data['repeated']

# สร้างตัวแปรสำหรับใช้งาน SMOTE
smote = SMOTE()

# ใช้งาน SMOTE เพื่อขยายข้อมูล
X_resampled, y_resampled = smote.fit_resample(X, y)

# สร้าง DataFrame ของข้อมูลที่ขยายแล้ว
resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.Series(y_resampled, name='ครั้งที่กระทำความผิด')], axis=1)

# บันทึกข้อมูลที่ขยายแล้วเป็นไฟล์ CSV
resampled_data.to_csv('smote_dataset.csv', index=False)
