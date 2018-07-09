"""
TODO : Write description
ResNet Module
"""

import math
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization


class ResNet():
    """
    TODO : Write description
    ResNet class

    input_shape is (w, h, ch)
    internal shape is (b, w, h, ch)
    """

    def __init__(self, input_shape, input_layers=None, channel_width=10, trainable=True):
        self.__trainable = trainable
        self.__input_shape = input_shape

        inputs = Input(self.__input_shape)
        if input_layers is not None:
            inputs = input_layers

        cv1 = self.__first_conv(inputs, 64, 7)
        mp1 = MaxPooling2D()(cv1)

        rb1 = self.__residual_block(mp1, 3, 64 * channel_width, is_first=True)
        rb2 = self.__residual_block(rb1, 4, 128 * channel_width)
        rb3 = self.__residual_block(rb2, 6, 256 * channel_width)
        rb4 = self.__residual_block(rb3, 3, 512 * channel_width)
        self.__network_without_head = rb4

        vp1 = AveragePooling2D(pool_size=7, strides=1, padding='same')(rb4)
        ft1 = Flatten()(vp1)
        outputs = Dense(1000, activation="softmax", trainable=self.__trainable)(ft1)
        self.__network = outputs

        self.__model = Model(inputs=[inputs], outputs=[outputs])


    def __first_conv(self, input_layer, output_channels, filter_size, strides=(2, 2)):
        conv = Conv2D(output_channels, filter_size, strides=strides, padding='same'
                      , input_shape=input_layer.get_shape()
                      , trainable=self.__trainable)(input_layer)
        norm = BatchNormalization()(conv)
        return Activation("relu")(norm)

    def __base_block(self, input_layer, output_channels, filter_size, first_strides=(1, 1)):
        deep_path = self.__deep_path(input_layer, output_channels, filter_size
                                     , first_strides=first_strides)
        shortcut = self.__shortcut(input_layer, deep_path)
        return Add()([shortcut, deep_path])

    def __deep_path(self, input_layer, output_channels, filter_size, first_strides=(1, 1)):
        output = input_layer
        hidden_channels = output_channels // 4
        if hidden_channels > 0:
            norm1 = BatchNormalization()(input_layer)
            rl1 = Activation("relu")(norm1)
            conv1 = Conv2D(hidden_channels, 1, trainable=self.__trainable)(rl1)
            norm2 = BatchNormalization()(conv1)
            rl2 = Activation("relu")(norm2)
            conv2 = Conv2D(hidden_channels, filter_size, strides=first_strides, padding='same'
                           , trainable=self.__trainable)(rl2)
            norm3 = BatchNormalization()(conv2)
            rl3 = Activation("relu")(norm3)
            do1 = Dropout(0.4)(rl3)
            conv3 = Conv2D(output_channels, 1, trainable=self.__trainable)(do1)
            output = conv3
        return output

    def __shortcut(self, input_layer, deep_path):
        input_shape = input_layer.get_shape().as_list()
        deep_path_shape = deep_path.get_shape().as_list()

        stride_w = math.ceil(input_shape[1] / deep_path_shape[1])
        stride_h = math.ceil(input_shape[2] / deep_path_shape[2])
        is_match_ch_num = input_shape[3] == deep_path_shape[3]

        shortcut = input_layer
        if stride_w > 1 or stride_h > 1 or not is_match_ch_num:
            shortcut = Conv2D(deep_path_shape[3], 1
                              , strides=(stride_w, stride_h)
                              , trainable=self.__trainable)(input_layer)
        return shortcut

    def __residual_block(self, input_layer, block_size, output_channels, is_first=False):
        output = input_layer
        for i in range(block_size):
            first_strides = (1, 1)
            if i == 0 and not is_first:
                first_strides = (2, 2)
            output = self.__base_block(input_layer, output_channels, 3, first_strides=first_strides)
        return output

    def get_input_size(self):
        """
        TODO : Write description
        get_input_size
        """
        return self.__input_shape

    def get_network(self, without_head=False):
        """
        TODO : Write description
        get_network
        """
        if without_head:
            return self.__network_without_head
        return self.__network

    def get_residual_network(self):
        """
        TODO : Write description
        get_residual_network
        """
        return self.get_network(without_head=True)

    def get_model(self):
        """
        TODO : Write description
        get_model
        """
        return self.__model

    def print_model_summay(self):
        """
        TODO : Write description
        print_model_summay
        """
        self.__model.summary()
