'''
Train a TableModel on a table of data.
'''

import json
import pickle

import numpy
import pandas
import yaml

from .. import table_model
from .. import table_classifier

import keras
import sklearn.metrics

class TrainedModel():
    '''
    Trained model is a loading and saving container for a TableModel and a
    KerasWideAndDeepClassifierModel. 
    '''
    def __init__(self, data_model, classifier_model):
        self.data_model = data_model
        self.classifier_model = classifier_model

    def predict(self, dict_from_json):
        '''
        Parameters
        ----------
        dict_from_json : dict
            Name value pairs that are a single sample to encode and predict.
        '''
        to_predict = self.data_model.transform_sample(dict_from_json)
        prediction, score = self.classifier_model.predict_score(to_predict)
        ordinal = numpy.min([prediction, len(self.data_model.classes) -1])
        return {'label': self.data_model.classes[ordinal], 'score': score}


    def __getstate__(self):
        return {
            'data_model': self.data_model,
            'classifier_config': self.classifier_model.model.get_config(),
            'classifier_weights': self.classifier_model.model.get_weights()
        } 

    def __setstate__(self, state):
        self.data_model = state['data_model']
        self.classifier_model = table_classifier.KerasWideAndDeepClassifierModel()
        self.classifier_model.model = keras.Model.from_config(state['classifier_config'])
        self.classifier_model.model.set_weights(state['classifier_weights'])


def train(arguments):
    '''
    Configure a model, load data, and train to fit.
    '''
    with open(arguments['<configuration_yaml>'], 'r', encoding='utf-8') as configuration_file:
        configuration = yaml.safe_load(configuration_file)
        data_model = table_model.TableModel.from_configuration(configuration)

    data = pandas.read_csv(arguments['<input_data_csv>'])
    x, y = data_model.fit_transform(data)

    classifier = table_classifier.KerasWideAndDeepClassifierModel()
    classifier.fit(x, y)

    trained_model = TrainedModel(data_model, classifier)

    # save out the trained model
    with open(arguments['<output_trained_model>'], 'wb') as model_file:
        pickle.dump(trained_model, model_file)
    
    # read the trained model and self check
    print('self check')
    with open(arguments['<output_trained_model>'], 'rb') as model_file:
        trained_model = pickle.load(model_file)
        one = json.loads(data.iloc[0].to_json(None))
        predictions = trained_model.predict(one)
        print('Predicting: ', one)
        print('Prediction', predictions)
        predictions = trained_model.classifier_model.predict(x)
        print('Accuracy {0:%}'.format(sklearn.metrics.accuracy_score(y, predictions)))
