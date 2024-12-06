# -*- coding:utf-8 -*-
# '''
# Created on 18-8-14 下午4:39
# 
# @Author: Greg Gao(laygin)
# '''
import os
from keras import backend as K
from keras.models import Model
from keras.engine.input_layer import Input
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
import numpy as np


def channel_split(x, name=''):
    # Equipartition
    in_channels = x.shape.as_list()[-1]
    ip = in_channels // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c


def channel_shuffle(x):
    # Shuffle channels in the tensor
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # Swap channels across 2nd and 4th dimensions
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):
    # Define batch norm axis for channels last format
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    # Prefix for layers naming
    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)

    # If strides < 2, split the channels and apply the shuffle unit
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    # 1x1 Conv to reduce dimensions
    x = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)

    # Depthwise 3x3 convolution
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)

    # 1x1 Conv again to reduce dimensions
    x = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    # Concatenate with shortcut connection (skip connection) if strides < 2
    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        # Apply a shortcut connection if strides == 2
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)

        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    # Apply channel shuffle operation
    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    # Apply shuffle unit for the first block
    x = shuffle_unit(x, out_channels=channel_map[stage-1], strides=2, bottleneck_ratio=bottleneck_ratio, stage=stage, block=1)

    # Apply shuffle units for the remaining blocks
    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1], strides=1, bottleneck_ratio=bottleneck_ratio, stage=stage, block=(1+i))

    return x

# Example usage
# if __name__ == '__main__':
#     from keras.utils import plot_model
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for testing purposes

#     # Example input shape and model configuration
#     input_shape = (224, 224, 3)
#     bottleneck_ratio = 1.0
#     num_shuffle_units = [3, 7, 3]
#     scale_factor = 1.0
#     channel_map = [24, 48, 96]  # Example channel map

#     # Create a dummy input tensor
#     img_input = Input(shape=input_shape)

#     # Build the model using blocks
#     x = Conv2D(24, kernel_size=(3, 3), padding='same', strides=2, activation='relu')(img_input)
#     for stage in range(len(num_shuffle_units)):
#         repeat = num_shuffle_units[stage]
#         x = block(x, channel_map, bottleneck_ratio=bottleneck_ratio, repeat=repeat, stage=stage + 2)

#     # Add global average pooling and output layer
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1000, activation='softmax')(x)

#     # Create the final model
#     model = Model(img_input, x, name="ShuffleNetV2")

#     # Plot the model architecture
#     plot_model(model, to_file='shufflenetv2.png', show_layer_names=True, show_shapes=True)

#     model.summary()
