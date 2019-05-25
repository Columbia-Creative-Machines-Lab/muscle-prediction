############################################ LSTM & time-series prediction

import keras
from keras.layers import LSTM
from matplotlib import pyplot as plt

class Model(object):
    def __init__(self):
        self.model = Sequential()
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
        self.model.add(Dense(32))
        self.model.add(Dropuot(0.5))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    """
    Set `overwrite=True` to overwrite all weights in `weight_file.hdf5`.
    """
    def train(self, name, train_X, train_y, val_X, val_y, overwrite=False, epochs=20, batch_size=10000):
        if not os.path.isfile('%s.hdf5' % name) or overwrite:
            history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(val_X, val_y), verbose=2, shuffle=True)
            save_model(self.model, name)
        else:
            model = load_model(name)
            history = None
        return model, history

    def plot_model(self, image_path = 'plots/model.png'):
        keras.utils.plot_model(self.model, show_shapes=True, to_file=image_path)
        plt.figure(dpi=150)
        imgplot = plt.imshow(plt.imread(image_path))
        plt.axis('off')
        plt.show()
