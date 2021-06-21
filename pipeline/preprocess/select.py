import pandas as pd


def select_satisfy_lipinski(X_train, y, X_test, id_test):
    X_train = X_train[X_train['Lipinski'] == 1].reset_index(drop=True)
    y = y.iloc[X_train.index.values].reset_index(drop=True)
    X_test = X_test[X_test['Lipinski'] == 1].reset_index(drop=True)
    id_test = id_test.iloc[X_test.index.values].reset_index(drop=True)
    return X_train, y, X_test, id_test


def select_not_satisfy_lipinski(X_train, y, X_test, id_test):
    X_train = X_train[X_train['Lipinski'] == 0].reset_index(drop=True)
    y = y.iloc[X_train.index.values].reset_index(drop=True)
    X_test = X_test[X_test['Lipinski'] == 0].reset_index(drop=True)
    id_test = id_test.iloc[X_test.index.values].reset_index(drop=True)
    return X_train, y, X_test, id_test


def select_core_feature(X_train, y, X_test, id_test):
    X_train = X_train[['Basicity', 'MW']]
    X_test = X_test[['Basicity', 'MW']]
    return X_train, y, X_test, id_test


def select_imp_features(df, n):
    df_imps = pd.read_csv('./models/17-featmie/LGB2-imps.csv')
    imps = df_imps.name.values
    return df[imps[:n]]


def select_imp_30_features(X_train, y, X_test, id_test):
    n = 30
    X_train = select_imp_features(X_train, n)
    X_test = select_imp_features(X_test, n)
    return X_train, y, X_test, id_test


def select_imp_60_features(X_train, y, X_test, id_test):
    n = 60
    X_train = select_imp_features(X_train, n)
    X_test = select_imp_features(X_test, n)
    return X_train, y, X_test, id_test


def select_imp_100_features(X_train, y, X_test, id_test):
    n = 100
    X_train = select_imp_features(X_train, n)
    X_test = select_imp_features(X_test, n)
    return X_train, y, X_test, id_test


def select_imp_300_features(X_train, y, X_test, id_test):
    n = 300
    X_train = select_imp_features(X_train, n)
    X_test = select_imp_features(X_test, n)
    return X_train, y, X_test, id_test


def select_imp_500_features(X_train, y, X_test, id_test):
    n = 500
    X_train = select_imp_features(X_train, n)
    X_test = select_imp_features(X_test, n)
    return X_train, y, X_test, id_test


def select_imp_800_features(X_train, y, X_test, id_test):
    n = 800
    X_train = select_imp_features(X_train, n)
    X_test = select_imp_features(X_test, n)
    return X_train, y, X_test, id_test


def select_imp_1000_features(X_train, y, X_test, id_test):
    n = 1000
    X_train = select_imp_features(X_train, n)
    X_test = select_imp_features(X_test, n)
    return X_train, y, X_test, id_test
