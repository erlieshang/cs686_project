# Author: erlie.shang@uwaterloo.ca
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.engine.topology import Layer
import pickle


class FactionalPooling(Layer):
    def __init__(self, pool_ratio=(1.44, 1.44), pseudo_random=None, overlapping=None, **kwargs):
        self.pool_ratio = [1.0]
        self.pool_ratio.extend(pool_ratio)
        self.pool_ratio.extend([1.0])
        self.pseudo_random = pseudo_random
        self.overlapping = overlapping
        self._ox = None
        self._oy = None
        super(FactionalPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FactionalPooling, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._ox, self._oy, input_shape[3])

    def call(self, x):
        output = tf.nn.fractional_max_pool(x, self.pool_ratio, self.pseudo_random, self.overlapping)
        self._ox = output[0].shape[1].value
        self._oy = output[0].shape[2].value
        return output[0]


class MixedPooling(Layer):
    def __init__(self, pool_size=(2, 2), strides=(1, 1), padding='VALID', **kwargs):
        self.pool_size = [1]
        self.pool_size.extend(pool_size)
        self.pool_size.extend([1])
        self.strides = [1]
        self.strides.extend(strides)
        self.strides.extend([1])
        self.padding = padding
        self._ox = None
        self._oy = None
        super(MixedPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='ratio', shape=(1,), initializer='uniform', trainable=True)
        super(MixedPooling, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._ox, self._oy, input_shape[3])

    def call(self, x):
        max_output = tf.nn.max_pool(x, self.pool_size, self.strides, self.padding)
        self._ox = max_output.shape[1].value
        self._oy = max_output.shape[2].value
        avg_output = tf.nn.avg_pool(x, self.pool_size, self.strides, self.padding)
        return self.kernel * max_output + (1 - self.kernel) * avg_output


class NNBase(object):
    def __init__(self, x_train, y_train, x_test, y_test, num_classes, epochs=200, batch_size=32, name='Default'):
        self.x_train = x_train.astype('float32') / 255
        self.y_train = keras.utils.to_categorical(y_train, num_classes)
        self.x_test = x_test.astype('float32') / 255
        self.y_test = keras.utils.to_categorical(y_test, num_classes)
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = name
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        return model

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                                 validation_data=(self.x_test, self.y_test))
        output = open(self.name + 'data.pkl', 'wb')
        pickle.dump([history.epoch, history.history], output)
        output.close()
        self.plot_history(history)

    def plot_history(self, history):
        plt.plot(history.epoch, history.history['val_acc'], color='blue', label='testing accuracy')
        plt.plot(history.epoch, history.history['acc'], color='red', label='training accuracy')
        plt.title('Accuracy Rate Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy Rate')
        plt.legend(loc='upper left', frameon=False)
        plt.grid()
        plt.savefig(self.name + '_accuracy.png')
        plt.close('all')
        plt.plot(history.epoch, history.history['val_loss'], color='blue', label='testing loss')
        plt.plot(history.epoch, history.history['loss'], color='red', label='training loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper left', frameon=False)
        plt.grid()
        plt.savefig(self.name + '_loss.png')
        plt.close('all')


class CNN(NNBase):
    def __init__(self, x_train, y_train, x_test, y_test, num_classes, epochs=200,
                 batch_size=32, pool_method='max', name='default'):
        self.pool_method = pool_method
        NNBase.__init__(self, x_train, y_train, x_test, y_test, num_classes, epochs, batch_size, name)

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        if self.pool_method == 'fmp':
            model.add(FactionalPooling(pool_ratio=(1.44, 1.44)))
        elif self.pool_method == 'max':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif self.pool_method == 'mixed':
            model.add(MixedPooling(pool_size=(2, 2)))
        else:
            assert False
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        if self.pool_method == 'fmp':
            model.add(FactionalPooling(pool_ratio=(1.44, 1.44)))
        elif self.pool_method == 'max':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif self.pool_method == 'mixed':
            model.add(MixedPooling(pool_size=(2, 2)))
        else:
            assert False
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    @staticmethod
    def fmp(x):
        output = tf.nn.fractional_max_pool(x, [1.0, 1.44, 1.44, 1.0], None, None)
        return output[0]


class DNN(NNBase):
    def __init__(self, x_train, y_train, x_test, y_test, num_classes, epochs=200, batch_size=32, name='default'):
        self.x_train = x_train.astype('float32') / 255
        self.y_train = keras.utils.to_categorical(y_train, num_classes)
        self.x_test = x_test.astype('float32') / 255
        self.y_test = keras.utils.to_categorical(y_test, num_classes)
        self.input_size = 1
        for size in x_train.shape[1:]:
            self.input_size *= size
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.input_size)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.input_size)
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = name
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_dim=self.input_size))
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[0:100]
    y_train = y_train[0:100]
    x_test = x_test[0:100]
    y_test = y_test[0:100]
    cnn = CNN(x_train, y_train, x_test, y_test, 10, pool_method='fmp', epochs=2, name='cnn_fmp')
    cnn.train()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # cnn = CNN(x_train, y_train, x_test, y_test, 10, epochs=2, name='cnn')
    # cnn.train()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # dnn = DNN(x_train, y_train, x_test, y_test, 10, epochs=100, name='dnn')
    # dnn.train()
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # cnn = CNN(x_train, y_train, x_test, y_test, 10, pool_method='mixed', epochs=5, name='cnn_mixed_trainable')
    # cnn.train()


if __name__ == "__main__":
    main()
