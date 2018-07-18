""" main """

import os
from matplotlib import pyplot
from keras.optimizers import SGD

from network.resnet import ResNet

INPUT_SHAPE = (2048, 2048, 3)
BATCH_SIZE = 256
EPOCHS = 100

DIR_MODEL = '.'
FILE_MODEL = 'ResNetModel.hdf5'

def train():
    """ train """
    print('execute train')

    # TODO
    train_inputs = None
    train_teachers = None
    test_inputs = None
    test_teachers = None

    network = ResNet(INPUT_SHAPE)
    network.print_model_summay()
    model = network.get_model()
    model.compile(optimizer=SGD(momentum=0.9, decay=0.0001)
                  , loss='categorical_crossentropy', metrics=['accuracy'])
    his = model.fit(train_inputs, train_teachers
                    , batch_size=BATCH_SIZE
                    , epochs=EPOCHS
                    , validation_data=(test_inputs, test_teachers)
                    , verbose=1)
    model.save_weights(os.path.join(DIR_MODEL, FILE_MODEL))
    plot_learning_curve(his)

def predict():
    """ predict """
    print('execute predict')

def plot_learning_curve(history):
    """ plot_learning_curve """
    x_axis = range(EPOCHS)
    pyplot.plot(x_axis, history.history['loss'], label="loss")
    pyplot.title("loss")
    pyplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pyplot.show()


if __name__ == '__main__':
    train()
    predict()
