import pandas as pd
from imblearn.over_sampling import SMOTE

# data = pd.read_csv(outdir+'/'+original_file_name)
data = pd.read_csv("../Data/clean/smote/raw_data/70_dataset.csv")

X = data.drop(['reci'], axis=1)
y = data['reci']

smote = SMOTE()

X_resampled, y_resampled = smote.fit_resample(X, y)

resampled_data = pd.concat([pd.DataFrame(X_resampled), pd.Series(
    y_resampled, name='reci')], axis=1)


resampled_data.to_csv('smote_dataset.csv', index=False)
