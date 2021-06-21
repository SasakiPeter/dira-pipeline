import pandas as pd
from pipeline.conf import settings


def load_train():
    path = settings.DATA_PATH['train']
    df = pd.read_csv(path)
    not_X = [settings.DATA_FORMAT['id'],
             settings.DATA_FORMAT['target']] + \
        settings.DATA_FORMAT['not_X']
    X = df.drop(not_X, axis=1)
    y = df[settings.DATA_FORMAT['target']]
    ID = df[settings.DATA_FORMAT['id']]
    return X, y, ID


def load_test():
    path = settings.DATA_PATH['test']
    df = pd.read_csv(path)
    not_X = [settings.DATA_FORMAT['id'],
             settings.DATA_FORMAT['target']] + \
        settings.DATA_FORMAT['not_X']
    X = df.drop(not_X, axis=1)
    ID = df[settings.DATA_FORMAT['id']]
    return X, ID
