'''
Table based data transformed for machine learning.
'''

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


class ExtractColumn(BaseEstimator, TransformerMixin):
    '''
    Pick a DataFrame apart by column.
    '''

    def __init__(self, column_or_series_name):
        '''
        Parameters
        ----------
        column_or_series_name : str
            Key into the data frame to extact a column or series.
        '''
        self._column_or_series_name = column_or_series_name

    def fit(self, X, y=None):
        '''
        No specific need to fit.
        '''
        return self

    def transform(self, data_frame):
        '''
        Parameters
        ----------
        data_frame : Pandas DataFrame
        '''
        return data_frame[self._column_or_series_name]
            

    def __repr__(self):
        '''
        Repr shows the column name for easier debugging.
        '''
        return "ExtractColumn('{0}')".format(self._column_or_series_name)


class PercentageColumn(BaseEstimator, TransformerMixin):
    '''
    Turn a string percentage column in a Pandas DataFrame like 12.5% into numerical values
    like 0.125.
    '''

    def to_float(self, X):
        '''
        Parameters
        ----------
        X : pandas series or numpy array
        '''
        def percent(x):
            '''
            A percentage as a number, or a NaN.
            '''
            if type(x) is str:
                return float(x.replace('%', '')) / 100.0
            elif x:
                return x / 100.0
            else:
                return 0

        return np.nan_to_num(np.array(X.map(percent)).reshape(-1, 1)).astype(np.float32)

    def fit(self, X, y=None):
        '''
        No specific need to fit.
        '''
        return self

    def transform(self, X):
        '''
        Transform a column of data into numerical percentage values.

        Parameters
        ----------
        X : pandas series or numpy array
        '''
        return self.to_float(X)


class NumericColumn(BaseEstimator, TransformerMixin):
    '''
    Take a numeric value column and standardize it.
    '''

    def __init__(self):
        '''
        Set up the internal transformation.
        '''
        self._transformer = MinMaxScaler()

    def fit(self, X, y=None):
        '''
        Fit the standardization.
        '''
        zeroed = pd.DataFrame(np.array(X).reshape(-1, 1)).fillna(0)
        self._transformer.fit(zeroed)
        return self

    def transform(self, X):
        '''
        Transform a column of data into numerical percentage values.

        Parameters
        ----------
        X : pandas series or numpy array
        '''
        zeroed = pd.DataFrame(np.array(X).reshape(-1, 1)).fillna(0)
        return self._transformer.transform(zeroed).astype(np.float32)


class CategoricalColumn(BaseEstimator, TransformerMixin):
    '''
    Take a string or key categorical column and transform it
    to one hot encodings.
    '''

    def __init__(self):
        '''
        Set up the internal transformation.
        '''
        self._labeler = LabelEncoder()
        self._encoder = OneHotEncoder()

    def fit(self, X, y=None):
        '''
        Fit the label and encoding
        '''
        handle_none = list(map(str, X))
        encoded = self._labeler.fit_transform(handle_none)
        self._encoder.fit(encoded.reshape(-1, 1))
        return self

    def transform(self, X):
        '''
        Transform a column of data into one hot encodings.

        Parameters
        ----------
        X : pandas series or numpy array
        '''
        handle_none = list(map(str, X))
        encoded = self._labeler.transform(handle_none)
        return self._encoder.transform(encoded.reshape(-1, 1)).todense().astype(np.float32)


class OutputLabelColumn(BaseEstimator, TransformerMixin):
    '''
    Take a string or key categorical column and transform it to integer labels.
    '''

    def __init__(self):
        '''
        Set up the internal transformation.
        '''
        self._labeler = LabelEncoder()

    def fit(self, X, y=None):
        '''
        Fit the label and encoding
        '''
        handle_none = list(map(str, X))
        self._labeler.fit(handle_none)
        return self

    def transform(self, X):
        '''
        Transform a column of data into one hot encodings.

        Parameters
        ----------
        X : pandas series or numpy array
        '''
        handle_none = list(map(str, X))
        return self._labeler.transform(handle_none).astype(np.int32)


class TableModel(BaseEstimator, TransformerMixin):
    '''
    Base TableModel, this deals with:
    * Specifying transformers at the column/series level for input
        (X) and output (Y) features
    * Combining transformed columns into a feature tensor
    * An override point to build a Keras model
    * Loading and Saving trained weights

    This is an in memory model, and will preserve the original DataFrame.

    >>> from tableclassifier import table_model
    >>> import pandas as pd
    >>> data = pd.DataFrame({'a': ['1%', '99%'], 'b': [1, 2], 'c': ['aa', 'bb'], 'd': [True, False]})
    >>> model = table_model.TableModel( \
            transformers={ \
                'a': table_model.PercentageColumn(), \
                'b': table_model.NumericColumn(), \
                'c': table_model.CategoricalColumn(), \
            }, \
            output_name='d' \
        )
    >>> X, Y = model.fit_transform(data)
    >>> X
    matrix([[ 0.01      ,  0.        ,  1.        ,  0.        ],
            [ 0.99000001,  1.        ,  0.        ,  1.        ]], dtype=float32)
    >>> Y
    matrix([[ 0.,  1.],
            [ 1.,  0.]], dtype=float32)
    '''

    def __init__(self, transformers={}, output_name=None):
        '''

        Initialize the model with column/series mappings, only those columns
        specified will be used in the model.

        Parameters
        ----------
        transformers : dict
            Mapping from column/series name to scikit learn style transformer
        output_name : str
            Name of a column/series that will be the output
        '''
        self.transformers = transformers.copy()
        self.output_name = output_name

    def fit(self, data_frame, y=None):
        '''
        Fit the column/series model based on the passed Pandas DataFrame.

        Parameters
        ----------
        data_frame : Pandas DataFrame
            Data frame containing both inputs and outputs in columns/series.
        '''
        if not hasattr(self, '_output'):
            # output is a categorical encoding column
            self._output = make_pipeline(
                ExtractColumn(self.output_name),
                OutputLabelColumn()
            )
            # each column is an extraction and then a transformation
            pipelines = [(name, make_pipeline(ExtractColumn(name), transformer))
                         for name, transformer in self.transformers.items()]
            self._input = FeatureUnion(pipelines)
        self._input.fit(data_frame)
        self._output.fit(data_frame)
        return self

    def transform(self, data_frame):
        '''
        Fit the column/series model based on the passed Pandas DataFrame.

        Parameters
        ----------
        data_frame : Pandas DataFrame
            Data frame containing both inputs and outputs in columns/series.
        '''
        return (
            self._input.transform(data_frame),
            self._output.transform(data_frame)
        )

    def transform_sample(self, data):
        '''
        Transform a single sample dictionary, which likely came in from JSON.
        '''
        data = pd.DataFrame([data])
        return self._input.transform(data)

    @property
    def classes(self):
        '''
        Returns
        -------
        An list of the class labels.
        '''
        return self._output.steps[1][1]._labeler.classes_.tolist()

    @classmethod
    def from_configuration(cls, config):
        '''
        Build up a table model from a configuration file, as passed in from YAML.
        >>> from tableclassifier import table_model
        >>> table_model.TableModel.from_configuration({'output': 'a', 'input': {'b': 'numeric', 'c': 'percentage', 'd': 'categorical'}})
        TableModel(output_name='a',
              transformers={'b': NumericColumn(), 'c': PercentageColumn(), 'd': CategoricalColumn()})
        '''
        column_map = {
            'numeric': NumericColumn,
            'percentage': PercentageColumn,
            'categorical': CategoricalColumn
        }
        input_config = {name: column_map[value]() for name, value in config['input'].items()}
        return cls(input_config, config['output'])
