import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

train = pd.read_csv("PATH_TO_YOUR_CLEAN_DF")

list_var_cat = train.select_dtypes('float').columns.tolist()
for i in list_var_cat:
    print('var', i, 'nb of modalities', train[i].nunique())
    le = LabelEncoder()
    train[i] = le.fit_transform(train[i])

print('TRAIN (df) SIZE BEFORE = ', train.shape)
train, test = train_test_split(train, test_size = 0.2)
print('TRAIN (df) SIZE AFTER = ', train.shape)

var_to_use = train.columns.tolist()
var_to_use.remove('NAME_OF_THE_COLUMN_TO_PREDICT')
print(f' Variables to use : {var_to_use}')

rf = RandomForestRegressor() #Use RandomForestClassifier() if your target is defined by classes.
                             # View potential parameters to improve your model.
rf.fit(train[var_to_use], train['NAME_OF_THE_COLUMN_TO_PREDICT'])
pred = rf.predict(test[var_to_use])
test['pred'] = pred
acc = accuracy_score(test['NAME_OF_THE_COLUMN_TO_PREDICT'], pred)

print(acc)  # Accuracy score
print(test) # The test df with the predictions displayed. 

# To save your model :
# joblib.dump(rf, "PATH/NAME")