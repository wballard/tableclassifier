'''
Serve a pretrained TableModel.
'''

import os
import pickle

import connexion

classify = None
DEBUG = True

def serve(arguments):
    '''
    Configure swagger, then mount and run a model. This will make use of a module
    level variable.
    '''
    with open(arguments['<trained_model>'], 'rb') as model_file:
        trained_model = pickle.load(model_file)
        def convert_numpy(dict_from_json):
            '''
            Make the prediction and deal with non-serializable numpy types.
            '''
            prediction = trained_model.predict(dict_from_json)
            prediction['score'] = float(prediction['score'])
            return prediction

        # expose the prediction method to connexion
        global classify
        classify = convert_numpy
    server_in = os.path.dirname(os.path.abspath(__file__))
    application = connexion.App(__name__, port=int(arguments['<port>']), specification_dir=server_in)
    application.add_api('api.yaml')
    application.run()
