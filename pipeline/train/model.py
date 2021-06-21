import os
import random
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import Input, Dense, Dropout, Activation


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def r2(y_true, y_pred):
    res = K.sum(K.square(y_true - y_pred))
    tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - res / tot


class NN(BaseEstimator, RegressorMixin):
    def __init__(self, num_layer=1, mid_units=300,
                 activation='relu', learning_rate=0.03, seed=0):
        self.num_layer = num_layer
        self.mid_units = mid_units
        self.activation = activation
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y, eval_set, fit_params={}):
        # def get_callback(name, params):
        #     callbacks = {
        #         'early_stopping': EarlyStopping
        #     }
        #     return callbacks[name](**params)

        # callbacks = fit_params['callbacks']
        # if callbacks:
        #     fit_params['callbacks'] =
        # [get_callback(name)for name in callbacks]
        K.clear_session()
        # tf.set_random_seed(self.seed)
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(self.seed)
        random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)
        fit_params = fit_params.copy()

        fit_params['x'] = X
        fit_params['y'] = y
        fit_params['validation_data'] = eval_set
        fit_params['callbacks'] = [EarlyStopping(
            patience=fit_params.pop('patience'),
            restore_best_weights=fit_params.pop('restore_best_weights')
        )]

        self.nBits_ = X.shape[1]
        self.model = self._create_model()
        self.model.compile(optimizer=Adam(
            lr=self.learning_rate), loss=root_mean_squared_error,
            metrics=[r2])
        self.model.fit(**fit_params)
        history = self.model.fit(**fit_params)
        self.best_iteration_ = len(
            history.history['val_loss']) - 50

    def predict(self, X):
        return self.model.predict(X).flatten()

    def _create_model(self):
        inputs = Input((self.nBits_, ))
        x = Dense(units=self.mid_units,
                  activation=self.activation, name='dense')(inputs)
        x = BatchNormalization(name='bn')(x)
        x = Activation(self.activation, name='activation')(x)
        x = Dropout(0.2, name='dropout')(x)
        for i in range(self.num_layer):
            x = Dense(units=self.mid_units, name=f'dense{i}')(x)
            x = BatchNormalization(name=f'bn{i}')(x)
            x = Activation(self.activation, name=f'activation{i}')(x)
            x = Dropout(0.2, name=f'dropout{i}')(x)
        outputs = Dense(units=1, activation='linear', name='output')(x)
        return Model(inputs=inputs, outputs=outputs)
