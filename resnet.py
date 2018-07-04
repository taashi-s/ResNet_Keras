from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Flatten, Dense, Reshape, Dropout, Activation
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization

class ResNet():
    def __init__(self, input_shape):
        inputs = Input(input_shape)

        cv1 = __first_conv(inputs, 64, 7)
        mp1 = MaxPooling2D()(cv1)

        rb1 = __residual_block(mp1, 3, 64, is_first=True)
        rb2 = __residual_block(rb1, 4, 128)
        rb3 = __residual_block(rb2, 6, 256)
        rb4 = __residual_block(rb3, 3, 512)

        vp1 = AveragePooling2D(pool_size=7, strides=1, padding='same')(rb4)
        ft1 = Flatten()(vp1)
        outputs = Dense(1000, activation="softmax")(ft1)

        self.MODEL = Model(inputs=[inputs], outputs=[outputs])

    def __first_conv(self, input, output_channels, filter_size, strides=(2, 2)):
        conv = Conv2D(output_channels, filter_size, strides=strides, padding='same',
                        input_shape=input.get_shape())(input)
        norm = BatchNormalization()(conv)
        return Activation("relu")(norm)

    def __base_block(self, input, output_channels, filter_size, first_strides=(1, 1)):
        deep_path = __deep_path(input, output_channels, filter_size, first_strides=strides)
        shortcut = __shortcut(input, deep_path)
        add = Add()([shortcut, deep_path])
        return Activation("relu")(add)

    def __deep_path(self, input, output_channels, filter_size, first_strides=(1, 1), with_last_Activation=False):
        output = input
        if 0 < output_channels / 4:
            conv1 = Conv2(output_channels / 4, 1)(input)
            norm1 = BatchNormalization()(conv1)
            rl1 = Activation("relu")(norm1)
            conv2 = Conv2(output_channels / 4, filter_size, stride=first_strides, padding='same')(conv1)
            norm2 = BatchNormalization()(conv2)
            rl2 = Activation("relu")(norm2)
            conv3 = Conv2(output_channels, 1)(rl2)
            norm3 = BatchNormalization()(conv3)
            output = norm3
            if with_last_Activation:
                output = Activation("relu")(norm3)
        return output

    def __shortcut(self, input, deep_path):
        input_shape = input.get_shape().as_list()
        deep_path_shape = deep_path.get_shape().as_list()

        stride_w = input_shape[2] / deep_path_shape[2]
        stride_h = input_shape[3] / deep_path_shape[3]
        is_match_ch_num = input_shape[1] == deep_path_shape[1]

        shortcut = input
        if 1 < stride_w or 1 < stride_h or not is_match_ch_num:
            shortcut = Conv2D(rdeep_path_shape[1], 1,
                                stride=(stride_w, stride_h))(input)
        return shortcut

    def __residual_block(self, input, block_size, output_channels, is_first=False):
        output = input
        for i in range(block_size):
            first_strides = (1, 1)
            if i == 0 and not is_first:
                first_strides = (2, 2)
            output = __base_block(input, output_channels, 3, first_strides=first_strides)
        return output

    def get_model(self):
        return self.model

