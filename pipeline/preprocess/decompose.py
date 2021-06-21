import pandas as pd
from sklearn.decomposition import PCA
from pipeline.preprocess.utils import get_binary_columns


def decompose_binary(X_train, y, X_test, id_test):
    n_components = 10
    columns = get_binary_columns(X_train)
    train_features = X_train.loc[:, columns]
    test_features = X_test.loc[:, columns]
    pca = PCA(n_components=n_components)
    pca.fit(train_features)
    X_train_ = X_train.drop(columns, axis=1)
    X_test_ = X_test.drop(columns, axis=1)
    columns = [f'PC{i+1}' for i in range(n_components)]
    X_train = pd.concat([X_train_, pd.DataFrame(
        pca.transform(train_features), columns=columns)], axis=1)
    X_test = pd.concat([X_test_, pd.DataFrame(
        pca.transform(test_features), columns=columns)], axis=1)
    return X_train, y, X_test, id_test
