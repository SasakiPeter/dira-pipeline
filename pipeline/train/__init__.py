import numpy as np
import pandas as pd

from pipeline.conf import settings
from pipeline.preprocess import load_train, load_test
from pipeline.train.base import CrossValidator
from pipeline.train.utils import (
    get_preprocess,
    get_transformer,
    get_eval_metric,
    get_split,
    get_model,
    get_oof_by_layer,
)
from pipeline.utils.directory import provide_dir


def train_by_layer(layer, X, y, id_train,
                   X_test, id_test, cv_summary, folder_path):
    original = X.copy(), y.copy(), X_test.copy(), id_test.copy()
    for name, params in layer.items():
        X, y, X_test, id_test = original

        if 'stratified' in params['CV'].keys():
            stratified_y = X.loc[:, params['CV']['stratified']]
            stratified = stratified_y >= np.median(stratified_y)
        else:
            stratified = None

        if params['PREPROCESS']:
            preprocess = params['PREPROCESS']
            preprocess_funcs = [get_preprocess(name) for name in preprocess]
            for f in preprocess_funcs:
                X, y, X_test, id_test = f(X, y, X_test, id_test)

        algo_name, section_id = name.split('_')
        eval_metrics = {metric: get_eval_metric(metric)
                        for metric in params['EVAL_METRICS']}

        # n_splits = params['CV']['n_splits']
        # seed = params['CV']['seed']

        if 'transform' in params.keys():
            transforms = [get_transformer(param)
                          for param in params['TRANSFORM']]
        else:
            transforms = None

        # params['CV']内に複数n_splitsがあればどうにかするやつ
        # section_idにシード値をつける必要がある？

        kf = get_split(algo_name, **params['CV'])
        cv = CrossValidator(get_model(algo_name, params['PARAMS']), kf)
        cv.run(
            X, y, id_train, X_test, id_test,
            eval_metrics=eval_metrics,
            prediction=params['PREDICT_FORMAT'],
            train_params={
                'cat_features': settings.CAT_IDXS,
                'fit_params': params['FIT_PARAMS']
            },
            transforms=transforms,
            stratified=stratified,
            verbose=1
        )
        models_path = f'{folder_path}/{section_id}.pkl'
        cv.save(models_path)
        cv_scores_path = f'{folder_path}/{section_id}.csv'
        cv.scores.to_csv(cv_scores_path, encoding='utf-8')
        cv_oof_path = f'{folder_path}/{section_id}-oof.csv'
        cv.save_oof(cv_oof_path)

        # feature importance
        image_path = f'{folder_path}/{section_id}.png'
        columns = X.columns.values
        cv.save_feature_importances(columns, image_path)
        csv_path = f'{folder_path}/{section_id}-imps.csv'
        cv.save_feature_importances_as_csv(columns, csv_path)

        for metric in eval_metrics.keys():
            mean = cv.scores.loc[metric, 'mean']
            sd = cv.scores.loc[metric, 'sd']
            se = cv.scores.loc[metric, 'se']
            ci = cv.scores.loc[metric, 'ci']
            cv_summary.loc[section_id, f'{metric}_mean'] = f'{mean:.5f}'
            cv_summary.loc[section_id, f'{metric}_sd'] = f'{sd:.5f}'
            cv_summary.loc[section_id, f'{metric}_se'] = f'{se:.5f}'
            cv_summary.loc[section_id, f'{metric}_ci'] = f'{ci:.5f}'
    return cv_summary


def train():
    # initialize
    folder_path = f'models/{settings.PROJECT_ID}'
    provide_dir(folder_path)
    cv_summary = pd.DataFrame()

    # loading data
    X, y, id_train = load_train()
    X_test, id_test = load_test()

    n_layers = settings.N_LAYERS

    # first layer
    first_layer = settings.FIRST_LAYER
    cv_summary = train_by_layer(
        first_layer, X, y, id_train, X_test, id_test, cv_summary, folder_path)

    # ここでseed averaging？
    # first layer内の選択したモデル群に対してアベレージングを行う
    # それぞれのモデルのスコアの平均値をとって、cv_summaryに加える

    # second layer
    if 2 <= n_layers:
        second_layer = settings.SECOND_LAYER
        X, X_test = get_oof_by_layer(first_layer)
        cv_summary = train_by_layer(
            second_layer, X, y, id_train, X_test, id_test, cv_summary, folder_path)

    path = f'{folder_path}/summary.csv'
    cv_summary.to_csv(path, encoding='utf-8')

    # X, X_test間違っているから、うまくいかないはず
    # Trainer, CrossValidatorに組み込む予定
    # blender = Blender()
    # blender.run(X, y, X_test, id_test, eval_metric=rmse)

    # path = f'{folder_path}/blend.csv'
    # blender.save_prediction(path)
    # with open(f'{folder_path}/blend.txt', 'w') as f:
    #     f.write(f'{blender.score}')

    # visualize的なmodule作った方がよさよう

    # import matplotlib.pyplot as plt
    # import numpy as np
    # columns = cv_summary.index.values

    # for i in range(cv_summary.columns.values.shape[0]//4):
    #     plt.figure(figsize=(12, 6))
    #     mean = cv_summary.iloc[:, i*4]
    #     se = cv_summary.iloc[:, i*4+2]
    #     metric, _ = mean.name.split('_')

    #     order = np.argsort(mean)
    #     plt.barh(np.array(columns)[order],
    #              mean[order], xerr=se[order])
    #     plt.xlabel('This error bar is SE')
    #     plt.savefig(f'{folder_path}/summary-{metric}.png')
    #     plt.close()
