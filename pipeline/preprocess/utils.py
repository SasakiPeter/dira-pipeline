import numpy as np


def get_object_columns(df):
    return [key for key, value in df.dtypes.items() if value == 'object']


def get_nan_columns(df):
    se = df.isnull().sum()
    return se[se > 0].index.values


def get_constant_columns(df):
    cols = df.columns.tolist()
    constants = []
    for c in cols:
        uniq = np.unique(df[c].values).shape[0]
        if uniq == 1:
            constants.append(c)
    return constants


def get_binary_columns(df):
    cols = df.columns.tolist()
    binary = []
    for c in cols:
        uniq = np.unique(df[c].values).shape[0]
        if uniq == 2:
            binary.append(c)
    return binary


# def extract_binary_columns(X_train, y, X_test, id_test):
#     df = pd.concat([X_train, X_test], axis=0)
#     columns = get_binary_columns(df)
#     X_train = X_train[columns]
#     X_test = X_test[columns]
#     return X_train, y, X_test, id_test
