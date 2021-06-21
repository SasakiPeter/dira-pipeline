import numpy as np
import pandas as pd

# def create_basicity(X_train, y, X_test, id_test):
#     X_train.loc[:, 'Basicity'] = X_train.loc[:, 'nBase'].values - \
#         X_train.loc[:, 'nAcid'].values
#     X_test.loc[:, 'Basicity'] = X_test.loc[:, 'nBase'].values - \
#         X_test.loc[:, 'nAcid'].values
#     return X_train, y, X_test, id_test


def basicity(df):
    return df.nBase - df.nAcid


def saturation1(df):
    return df.nH/((df.nC+df.nO)*2+1)


def saturation2(df):
    return df.nH/((df.nC+df.nO+df.nS)*2+df.nN+df.nP+1)


def log_feat1(df):
    return np.log((df.nN+df.nO+df.nS+df.nF+df.nCl +
                   df.nSpiro+df.nBridgehead)/df.nAtom)


def log_phosphate(df):
    return np.log(df.nO/(df.nS*3+0.1)+0.1)


def log_amide(df):
    return np.log(df.nO/(df.nN+0.1)+0.1)


def ring_distortion_score(df):
    energy = {
        '3': 27.5,
        '4': 26.3,
        '5': 6.2,
        '6': 0.1,
        '7': 6.2,
        '8': 9.7,
        '9': 12.6,
        '10': 12.4,
        '11': 11.3,
        '12': 4.1,
        'G12': 2.0
    }
    retval = np.zeros(df.shape[0])
    for k, v in energy.items():
        try:
            retval += df[f'n{k}Ring'].values * v
        except KeyError:
            pass
    return np.log(retval+0.1)


def aring_distortion_score_by_natom(df):
    energy = {
        '3': 27.5,
        '4': 26.3,
        '5': 6.2,
        '6': 0.1,
        '7': 6.2,
        '8': 9.7,
        '9': 12.6,
        '10': 12.4,
        '11': 11.3,
        '12': 4.1,
        'G12': 2.0
    }
    retval = np.zeros(df.shape[0])
    for k, v in energy.items():
        try:
            retval += df[f'n{k}ARing'].values * v
        except KeyError:
            pass
    return np.log(retval/df.nAtom+0.1)


def fring_distortion_score_by_natom(df):
    tri = (3**.5)/4
    squ = 1
    pen = (25 + 10*(5**.5))**.5/4
    hexa = 3*(3**.5)/2
    area = {
        '4': tri + tri,
        '5': tri + squ,
        '6': squ + squ,
        '7':  squ + pen,
        '8': pen + pen,
        '9': pen + hexa,
        '10': hexa + hexa,
        '11': pen + pen + pen,
        '12': pen + pen + hexa,
        'G12': pen + hexa + hexa
    }
    retval = np.zeros(df.shape[0])
    for k, v in area.items():
        try:
            retval += df[f'n{k}FRing'].values * v
        except KeyError:
            pass
    return np.log(retval/df.nAtom+0.1)


def o_rate(df):
    return np.log(df.nO / df.WPath * 1000 + 1)


def create_features2(X_train, y, X_test, id_test):
    X_train.loc[:, 'Ring_Dis'] = ring_distortion_score(X_train)
    X_test.loc[:, 'Ring_Dis'] = ring_distortion_score(X_test)
    X_train.loc[:, 'ARing_Dis'] = aring_distortion_score_by_natom(X_train)
    X_test.loc[:, 'ARing_Dis'] = aring_distortion_score_by_natom(X_test)
    X_train.loc[:, 'FRing_Dis'] = fring_distortion_score_by_natom(X_train)
    X_test.loc[:, 'FRing_Dis'] = fring_distortion_score_by_natom(X_test)
    X_train.loc[:, 'O_rate'] = o_rate(X_train)
    X_test.loc[:, 'O_rate'] = o_rate(X_test)
    return X_train, y, X_test, id_test


def create_features(X_train, y, X_test, id_test):
    X_train.loc[:, 'Basicity'] = basicity(X_train)
    X_train.loc[:, 'Satu1'] = saturation1(X_train)
    X_train.loc[:, 'Satu2'] = saturation2(X_train)
    X_train.loc[:, 'Log_feat1'] = log_feat1(X_train)
    X_train.loc[:, 'Log_phos'] = log_phosphate(X_train)
    X_train.loc[:, 'Log_amide'] = log_amide(X_train)
    X_test.loc[:, 'Basicity'] = basicity(X_test)
    X_test.loc[:, 'Satu1'] = saturation1(X_test)
    X_test.loc[:, 'Satu2'] = saturation2(X_test)
    X_test.loc[:, 'Log_feat1'] = log_feat1(X_test)
    X_test.loc[:, 'Log_phos'] = log_phosphate(X_test)
    X_test.loc[:, 'Log_amide'] = log_amide(X_test)
    return X_train, y, X_test, id_test


def join_mies(X_train, y, X_test, id_test):
    train_feat = pd.read_csv(
        './data/result_foo-train.csv').drop(['SMILES'], axis=1)
    test_feat = pd.read_csv(
        './data/result_foo-test.csv').drop(['SMILES'], axis=1)
    X_train = pd.concat([X_train, train_feat], axis=1)
    X_test = pd.concat([X_test, test_feat], axis=1)
    return X_train, y, X_test, id_test
