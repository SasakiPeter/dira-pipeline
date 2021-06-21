from sklearn.preprocessing import StandardScaler, QuantileTransformer


def standardize(X_train, y, X_test, id_test):
    scaler = StandardScaler()
    X_train.iloc[:, :] = scaler.fit_transform(X_train)
    X_test.iloc[:, :] = scaler.transform(X_test)
    return X_train, y, X_test, id_test


def rankgauss(X_train, y, X_test, id_test):
    rg = QuantileTransformer(
        n_quantiles=100,
        random_state=0,
        output_distribution='normal'
    )
    X_train.iloc[:, :] = rg.fit_transform(X_train)
    X_test.iloc[:, :] = rg.transform(X_test)
    return X_train, y, X_test, id_test

# def rank_gauss(x):
#     from scipy.special import erfinv
#     N = x.shape[0]
#     temp = x.argsort()
#     rank_x = temp.argsort() / N
#     rank_x -= rank_x.mean()
#     rank_x *= 2
#     efi_x = erfinv(rank_x)
#     efi_x -= efi_x.mean()
#     return efi_x

# for i in X.columns:
#     #print('Categorical: ',i)
#     X[i] = rank_gauss(X[i].values)


# def standardization(X):
#     return (X - X.mean()) / X.std(ddof=1)


# def standardize(X_train, X_valid, y_train, y_valid, X_test):
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_valid = scaler.fit_transform(X_valid)
#     X_test = scaler.transform(X_test)
#     return X_train, X_valid, y_train, y_valid, X_test

    # idx = X_train.shape[0]
    # X = pd.concat([X_train, X_test])
    # X = standardization(X)
    # X_train = X.iloc[:idx, :]
    # X_test = X.iloc[idx:, :]
    # return X_train, y, X_test, id_test
