'''
Train a TableModel on a table of data.
'''

import yaml


def train(arguments):
    '''
    Configure a model, load data, and train to fit.
    '''
    with open(arguments['<configuration_yaml>'], 'r', encoding='utf-8') as configuration_file:
        configuration = yaml.safe_load(configuration_file)
        print(configuration)


