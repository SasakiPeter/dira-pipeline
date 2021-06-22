import pandas as pd
from pipeline.conf import settings
from pipeline.train.base import CrossValidator


from pipeline.preprocess import (
    label_encoding, nan_to_zero, remove_const,
    select_satisfy_lipinski, select_not_satisfy_lipinski,
    create_features, create_features2,
    remove_nan, select_core_feature,
    decompose_binary, standardize, rankgauss, join_mies,
    select_imp_30_features,
    select_imp_60_features,
    select_imp_100_features,
    select_imp_300_features,
    select_imp_500_features,
    select_imp_800_features,
    select_imp_1000_features
)

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, log_loss,
    r2_score
)
from pipeline.utils.metrics import rmse

from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso)
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from pipeline.train.model import NN


def get_preprocess(name):
    preprocesses = {
        'LabelEncoding': label_encoding,
        'NanToZero': nan_to_zero,
        'RemoveConst': remove_const,
        'SelectSatisfyLipinski': select_satisfy_lipinski,
        'SelectNotSatisfyLipinski': select_not_satisfy_lipinski,
        # 'CreateBasicity': create_basicity,
        'CreateFeatures': create_features,
        'CreateFeatures2': create_features2,
        'JoinMies': join_mies,
        'RemoveNan': remove_nan,
        'SelectCoreFeature': select_core_feature,
        'DecomposeBinary': decompose_binary,
        'Standardize': standardize,
        'RankGauss': rankgauss,
        'SelectImp30Features': select_imp_30_features,
        'SelectImp60Features': select_imp_60_features,
        'SelectImp100Features': select_imp_100_features,
        'SelectImp300Features': select_imp_300_features,
        'SelectImp500Features': select_imp_500_features,
        'SelectImp800Features': select_imp_800_features,
        'SelectImp1000Features': select_imp_1000_features,
    }
    return preprocesses[name]


def get_transformer(name):
    transformer = {
        # 'DecomposeBinary': decompose_binary,
        # 'Standardize': standardize,
    }
    return transformer[name]


def get_split(algo, n_splits, seed, stratified=None):
    regression = {
        'CatBoostRegressor',
        'LGBMRegressor',
        'RandomForestRegressor',
        'LinearRegression',
        'Ridge', 'Lasso',
        'SVR', 'NN'
    }
    classification = {
        'CatBoostClassifier',
        'LGBMClassifier',
        'RandomForestClassifier',
        'LogisticRegression',
        'SVC',
    }
    if algo in regression:
        if stratified:
            return StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
        else:
            return KFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
    elif algo in classification:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    else:
        raise NotImplementedError(
            f'Trainer class must provide a {algo} model')


def multi_auc(y_true, y_pred):
    nunique = len(y_true.unique())
    roc = {idx: 0 for idx in range(nunique)}
    for idx in range(nunique):
        roc[idx] += roc_auc_score(y_true.apply(
            lambda x: 1 if x == idx else 0), y_pred[:, idx])
    score = sum(val for val in roc.values()) / nunique
    return score


def get_eval_metric(name):
    eval_metrics = {
        'RMSE': rmse,
        'R2': r2_score,
        'logloss': log_loss,
        'AUC': roc_auc_score,
        'multi_auc': multi_auc
    }
    return eval_metrics[name]


def get_model(name, params):
    models = {
        'CatBoostRegressor': CatBoostRegressor,
        'CatBoostClassifier': CatBoostClassifier,
        'LGBMRegressor': LGBMRegressor,
        'LGBMClassifier': LGBMClassifier,
        'RandomForestRegressor': RandomForestRegressor,
        'RandomForestClassifier': RandomForestClassifier,
        'LinearRegression': LinearRegression,
        'LogisticRegression': LogisticRegression,
        'Ridge': Ridge,
        'Lasso': Lasso,
        'SVR': SVR,
        'SVC': SVC,
        'NN': NN,
    }
    return models[name](**params)


def get_cvs_by_layer(layer):
    cvs = {}
    for name, params in layer.items():
        algo_name, section_id = name.split('_')
        model_path = f'models/{settings.PROJECT_ID}/{section_id}.pkl'
        cv = CrossValidator()
        cv.load(model_path)
        cvs[section_id] = cv
    return cvs


def get_oof_by_layer(layer):
    X = pd.DataFrame()
    X_test = pd.DataFrame()
    cvs = get_cvs_by_layer(layer)
    for section_id, cv in cvs.items():
        X[section_id] = cv.oof
        X_test[section_id] = cv.pred
    return X, X_test
