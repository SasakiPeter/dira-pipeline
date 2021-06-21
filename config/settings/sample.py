PROJECT_ID = 'sample'


# Data source definition
DATA_PATH = {
    'train': 'data/train.csv',
    'test': 'data/test.csv'
}

# semi target っていうかtarget複数設定して、それぞれの
# 学習器に応じて、切り替えられるように実装したい
DATA_FORMAT = {
    'id': 'CID',
    'target': 'y',
    'not_X': ['IsomericSMILES', 'name', 'split'],
}

CAT_IDXS = []


# Training


N_LAYERS = 2

FIRST_LAYER = {
    # 'NN_NN-1': {
    #     'PREPROCESS': [
    #         'NanToZero',
    #         'RemoveConst',
    #         'Standardize',
    #     ],
    #     # 'TRANSFORM': [
    #     #     'Standardize',
    #     # ],
    #     'CV': {
    #         'n_splits': 5,
    #         'seed': 1,
    #         # 'stratified': 'Lipinski'
    #     },
    #     'PARAMS': {
    #         'num_layer': 4,
    #         'mid_units': 485,
    #         'activation': 'relu',
    #         'learning_rate': 0.001,
    #         'seed': 2
    #     },
    #     'FIT_PARAMS': {
    #         'verbose': 0,
    #         'epochs': 1000,
    #         'batch_size': 100,
    #         # 'callbacks': ['early_stopping'],
    #         'patience': 50,
    #         'restore_best_weights': True
    #     },
    #     # 'EARLY_STOPPING': {
    #     # },
    #     'EVAL_METRICS': [
    #         'RMSE',
    #         'R2'
    #     ],
    #     'PREDICT_FORMAT': 'predict'
    # },
    'LGBMRegressor_LGB1': {
        'PREPROCESS': [
            'RemoveConst'
        ],
        'CV': {
            'n_splits': 2,
            'seed': 1,
        },
        'PARAMS': {
            'objective': 'regression',
            'boosting': 'gbdt',
            'tree_learner': 'serial',
            'nthread': -1,
            'seed': 0,

            'num_leaves': 63,
            'min_data_in_leaf': 20,
            'max_depth': 7,

            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'bagging_seed': 0,
            # 'feature_fraction': 1

            'save_binary': True,

            'max_bin': 255,
            'learning_rate': 0.1,

            'min_sum_hessian_in_leaf': 0.1,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'min_gain_to_split': 0.0,

            'verbose': -1,
            # 'metric': 'cross_entropy',
            'metric': 'rmse',
            'histogram_pool_size': 1024,
            'n_estimators': 10000,
        },
        'FIT_PARAMS': {
            'early_stopping_rounds': 100,
        },
        'EVAL_METRICS': [
            'RMSE',
            'R2'
            # 'AUC',
            # 'logloss'
        ],
        'PREDICT_FORMAT': 'predict'
    },
    'CatBoostRegressor_CTB1': {
        'PREPROCESS': [
            'RemoveConst'
        ],
        'CV': {
            'n_splits': 2,
            'seed': 1,
            # 'stratified': 'Lipinski',
        },
        'PARAMS': {
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 608,
            'learning_rate': 0.1,

            'bootstrap_type': 'Bayesian',
            'sampling_frequency': 'PerTreeLevel',
            'sampling_unit': 'Object',

            # up to 16
            'depth': 8,
            # try diff value
            'l2_leaf_reg': 3.0,
            # 'random_strength': 1,
            'bagging_temperature': 0,
            'border_count': 254,

            # golden feature
            # 'per_float_feature_quantization': '0:border_count=1024'

            'grow_policy': 'SymmetricTree',
            'nan_mode': 'Min',
            # 陽性の重みを増やす
            # 'scale_pos_weight': 9,
            'iterations': 10000,
        },
        'FIT_PARAMS': {
            'early_stopping_rounds': 100,
        },
        'EVAL_METRICS': [
            'RMSE',
            'R2'
        ],
        'PREDICT_FORMAT': 'predict'
    },
}

SECOND_LAYER = {
    'LinearRegression_STK1': {
        'PREPROCESS': [],
        'CV': {
            'n_splits': 2,
            'seed': 10
        },
        'PARAMS': {

        },
        'FIT_PARAMS': {

        },
        'EVAL_METRICS': [
            # 'AUC',
            # 'logloss'
            'RMSE',
            'R2'
        ],
        'PREDICT_FORMAT': 'predict'
    },
    # 'Ridge_STK2': {
    #     'PREPROCESS': [],
    #     'CV': {
    #         'n_splits': 2,
    #         'seed': 10
    #     },
    #     'PARAMS': {
    #         'alpha': 1.0,
    #         'random_state': 1
    #     },
    #     'FIT_PARAMS': {

    #     },
    #     'EVAL_METRICS': [
    #         # 'RMSE',
    #         # 'R2'
    #         'AUC',
    #         'logloss'
    #     ],
    #     'PREDICT_FORMAT': 'predict'
    # },
}
