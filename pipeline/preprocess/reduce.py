import pandas as pd
from pipeline.preprocess.utils import get_nan_columns, get_constant_columns


def remove_const(X_train, y, X_test, id_test):
    df = pd.concat([X_train, X_test], axis=0)
    columns = get_constant_columns(df)
    X_train.drop(columns, axis=1, inplace=True)
    X_test.drop(columns, axis=1, inplace=True)
    return X_train, y, X_test, id_test


def remove_nan(X_train, y, X_test, id_test):
    df = pd.concat([X_train, X_test], axis=0)
    columns = get_nan_columns(df)
    X_train.drop(columns, axis=1, inplace=True)
    X_test.drop(columns, axis=1, inplace=True)
    return X_train, y, X_test, id_test
