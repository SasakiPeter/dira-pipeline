import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def nan_to_zero(X_train, y, X_test, id_test):
    imp = SimpleImputer(missing_values=np.nan,
                        strategy='constant', fill_value=0)
    train_columns = X_train.columns.values
    test_columns = X_test.columns.values
    X_train = imp.fit_transform(X_train)
    X_test = imp.fit_transform(X_test)
    X_train = pd.DataFrame(X_train, columns=train_columns)
    X_test = pd.DataFrame(X_test, columns=test_columns)
    return X_train, y, X_test, id_test
