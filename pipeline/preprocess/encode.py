import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pipeline.preprocess.utils import get_object_columns


def label_encoding(X_train, y, X_test, id_test):
    df = pd.concat([X_train, X_test], axis=0)
    columns = get_object_columns(df)
    for col in columns:
        le = LabelEncoder()
        le.fit(df[col].values)
        X_train[col] = le.transform(X_train[col].values)
        X_test[col] = le.transform(X_test[col].values)
    return X_train, y, X_test, id_test
