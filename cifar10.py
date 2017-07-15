# Author: erlie.shang@uwaterloo.ca
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.engine.topology import Layer
import tensorflow as tf
import matplotlib.pyplot as plt


class FractionalMaxPooling(Layer):
    def __init__(self, pooling_ratio, pseudo_random=None, overlapping=None, **kwargs):
        super(FractionalMaxPooling, self).__init__(**kwargs)
        self.pooling_ratio = [1.0, pooling_ratio, pooling_ratio, 1.0]
        self.pseudo_random = pseudo_random
        self.overlapping = overlapping
        self.output_x = None
        self.output_y = None

    def call(self, x):
        output = tf.nn.fractional_max_pool(x, self.pooling_ratio, self.pseudo_random, self.overlapping)
        self.output_x = output[0].shape[1].value
        self.output_y = output[0].shape[2].value
        return output[0]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_x, self.output_y, input_shape[3])


class NNBase(object):
    def __init__(self, x_train, y_train, x_test, y_test, num_classes, epochs=200, data_augmentation=True,
                 batch_size=32, name='Default'):
        self.x_train = x_train.astype('float32') / 255
        self.y_train = keras.utils.to_categorical(y_train, num_classes)
        self.x_test = x_test.astype('float32') / 255
        self.y_test = keras.utils.to_categorical(y_test, num_classes)
        self.num_classes = num_classes
        self.epochs = epochs
        self.data_aug = data_augmentation
        self.batch_size = batch_size
        self.name = name
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        return model

    def train(self):
        if not self.data_aug:
            print('Not using data augmentation.')
            history = self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                                     validation_data=(self.x_test, self.y_test))
            plt.plot(history.epoch, history.history['val_acc'], color='blue', label='testing accuracy')
            plt.plot(history.epoch, history.history['acc'], color='red', label='training accuracy')
            plt.title('Accuracy Rate Curve')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy Rate')
            plt.grid()
            plt.savefig(self.name + '_accuracy.png')
            plt.close('all')
            plt.plot(history.epoch, history.history['val_loss'], color='blue', label='testing loss')
            plt.plot(history.epoch, history.history['loss'], color='red', label='training loss')
            plt.title('Loss Curve')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid()
            plt.savefig(self.name + '_loss.png')
            plt.close('all')

        else:
            print('Using real-time data augmentation.')
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=False)

            datagen.fit(self.x_train)
            self.model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size),
                                     steps_per_epoch=self.x_train.shape[0] // self.batch_size, epochs=self.epochs,
                                     validation_data=(self.x_test, self.y_test))


class CNN(NNBase):
    def __init__(self, x_train, y_train, x_test, y_test, num_classes, epochs=200, data_augmentation=True,
                 batch_size=32, use_fmp=False, name='default'):
        self.use_fmp = use_fmp
        NNBase.__init__(self, x_train, y_train, x_test, y_test, num_classes, epochs, data_augmentation, batch_size, name)

    def _build_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.x_train.shape[1:], activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        if self.use_fmp:
            model.add(FractionalMaxPooling(1.44))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        if self.use_fmp:
            model.add(FractionalMaxPooling(1.44))
        else:
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.num_classes, activation='softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model


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
        self.data_aug = False
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
    cnn = CNN(x_train, y_train, x_test, y_test, 10, use_fmp=True, epochs=200, name='cnn_fmp')
    cnn.train()
    dnn = DNN(x_train, y_train, x_test, y_test, 10, epochs=200, name='dnn')
    dnn.train()


if __name__ == "__main__":
    main()
