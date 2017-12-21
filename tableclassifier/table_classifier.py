'''
Keras models for learning classification on table data
'''

import keras
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class KerasClassifierModel(BaseEstimator, ClassifierMixin):
    '''
    Base class for Keras classification models.
    '''

    def __init__(self, verbose=1, epochs=256, batch_size=512):
        '''
        Parameters
        ----------
        verbose : boolean
            Show more output
        epochs : int
            Train this number of cycles
        batch_size : int
            This number of samples per batch
        '''
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size

    def compute_class_weights(self, values):
        '''
        Compute the class weighting dictionary for use with
        imbalanced classes.

        Parameters
        ----------
        values : 1D numpy array of values
        '''
        values = np.ravel(values)
        labels, counts = np.unique(values, return_counts=True)
        if len(labels) == 1:
            return None
        class_weight={
            label: len(values) / count
            for (label, count) in zip(labels, counts)
        }
        return class_weight

    def predict(self, x, batch_size=32, verbose=0):
        '''
        Generate class predictions for the input samples.
        The input samples are processed batch by batch.
        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.
        # Returns
            A numpy array of class predictions.
        '''
        proba = self.model.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')


class KerasWideAndDeepClassifierModel(KerasClassifierModel):
    '''
    Wide and Deep implemented with Keras.
    '''

    def fit(self, x, y):
        '''
        Create and fit a logistic regression model

        Parameters
        ----------
        x : 2d numpy array
            0 dimension is batch, 1 dimension features
        y : 1d numpy array
            each entry is a class label
        '''
        class_weight = self.compute_class_weights(y)
        print('class weight', class_weight)
        y = keras.utils.to_categorical(y).astype(np.float32)

        HIDDEN = 512

        deep = keras.models.Sequential()
        deep.add(keras.layers.Dense(HIDDEN, activation='relu', input_dim=x.shape[1]))
        deep.add(keras.layers.BatchNormalization())
        deep.add(keras.layers.Dense(HIDDEN, activation='relu'))
        deep.add(keras.layers.BatchNormalization())
        deep.add(keras.layers.Dense(HIDDEN//2, activation='relu'))
        deep.add(keras.layers.BatchNormalization())
        deep.add(keras.layers.Dense(HIDDEN//2, activation='relu'))
        deep.add(keras.layers.BatchNormalization())
        deep.add(keras.layers.Dense(y.shape[1], activation='sigmoid'))

        wide = keras.models.Sequential()
        wide.add(keras.layers.Dense(
            y.shape[1], activation='sigmoid', input_dim=x.shape[1]))

        input = keras.layers.Input(shape=(x.shape[1],))
        wide = wide(input)
        deep = deep(input)
        wide_deep = keras.layers.Add()([wide, deep])
        output = keras.layers.Dense(y.shape[1], activation = 'sigmoid')(wide_deep)

        model=keras.models.Model(inputs = [input], outputs = [output])

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
        self.model=model
        model.fit(x, y,
                  epochs = self.epochs,
                  batch_size = self.batch_size,
                  verbose = self.verbose,
                  class_weight = class_weight,
                  callbacks = [keras.callbacks.EarlyStopping(monitor='loss', patience=4)])
        return self
