############################################ LSTM & time-series prediction

import keras
import os
from matplotlib import pyplot as plt
import keras
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential, model_from_json
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools

class Model(object):

    def __init__(self, name, input_shape=(0,0)):
        assert len(input_shape)==2
        self.history = None
        self.name = name 
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=input_shape))
        self.model.add(Dense(32))
        self.model.add(Dropout(0.3)) # 0.5
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def _save_model(self, name='X61'):
        model_json = self.model.to_json()
        with open('%s.json' % name, 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights('%s.h5' % name)
        self.model.save('%s.hdf5' % name)

    def _load_model(self, name='X61'):
        json_file = open('%s.json' % name, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('%s.h5' % name)
        return loaded_model

    """
    Set `overwrite=True` to overwrite all weights in `weight_file.hdf5`.
    """
    def train(self, train_X, train_y, val_X, val_y, name='X61', overwrite=False, epochs=20, batch_size=10000):
        history = None
        if overwrite or not os.path.isfile('%s.hdf5' % name):
            self.history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=2, shuffle=True)
            self._save_model(name)
        else:
            self.model = self._load_model(name)

    def predict(self, X, y=None, plot=True):
        yhat = self.model.predict(X)
        if plot:
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
            ax.plot(yhat, alpha=0.8, c='r', label='Predicted force')
            if 'numpy' in str(type(y)):
                ax.plot(y, alpha=0.3, c='b', label='Measured force')
            ax.set_title('Prediction on test data')
            ax.set_xlabel('Time step')
            ax.set_ylabel('Force (N)')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig('plots/%s_test_prediction' % self.name, dpi=300)
            plt.show()
        return yhat

    def plot_model(self, image_path='plots/model.png'):
        keras.utils.plot_model(self.model, show_shapes=True, to_file=image_path)
        plt.figure(dpi=150)
        imgplot = plt.imshow(plt.imread(image_path))
        plt.axis('off')
        plt.show()

    def plot_history(self, n_epochs=20, n_cycles=100):
        if self.history:
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('LSTM loss (MSE) on %d epochs, %d cycles' % (n_epochs, n_cycles))
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper right')
            plt.show()
        plt.savefig('plots/%s_loss' % self.name)
