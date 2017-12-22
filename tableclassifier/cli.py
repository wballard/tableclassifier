'''tableclassifier

Usage:
    tableclassifier train <configuration_yaml> <input_data_csv> <output_trained_model>
    tableclassifier serve <trained_model> <port>
'''

import docopt
from . import server
from . import trainer

def execute():
    arguments = docopt.docopt(__doc__)    
    if arguments['train']:
        trainer.train(arguments)
    if arguments['serve']:
        server.serve(arguments)
