'''
Train a TableModel on a table of data.
'''

import json
import pickle

import pandas
import yaml

from .. import table_model
from .. import table_classifier

import keras


def train(arguments):
    '''
    Configure a model, load data, and train to fit.
    '''
    with open(arguments['<configuration_yaml>'], 'r', encoding='utf-8') as configuration_file:
        configuration = yaml.safe_load(configuration_file)
        data_model = table_model.TableModel.from_configuration(configuration)
        print(data_model)

    data = pandas.read_csv(arguments['<input_data_csv>'])
    x, y = data_model.fit_transform(data)
    print(x)
    print(y)

    classifier = table_classifier.KerasWideAndDeepClassifierModel()
    classifier.fit(x, y)

    classifier_model_config = classifier.model.get_config()
    classifier_model_weights = classifier.model.get_weights()

    # save out the trained model
    with open(arguments['<output_trained_model>'], 'wb') as model_file:
        pickle.dump(data_model, model_file)
    
    with open(arguments['<output_trained_model>'], 'rb') as model_file:
        data_model = pickle.load(model_file)
        data.drop('loan_status', axis=1, inplace=True)
        print(data_model)
        print(data_model.classes)
        one = json.loads(data.iloc[0].to_json(None))
        print(one)
        print(data_model.transform_sample(one))
        classifier = table_classifier.KerasWideAndDeepClassifierModel(epochs=1)
        classifier.model = keras.Model.from_config(classifier_model_config)
        classifier.model.set_weights(classifier_model_weights)
        predictions = classifier.predict(one)
        print(predictions)
