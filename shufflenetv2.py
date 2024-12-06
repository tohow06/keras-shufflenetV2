# -*- coding:utf-8 -*-
# '''
# Created on 18-8-14 下午4:48
# 
# @Author: Greg Gao(laygin)
# '''
import numpy as np
from keras.utils import plot_model
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dense
from keras.models import Model
import keras.backend as K
from utils import block

def get_input_shape(input_shape, default_size=224, min_size=28, require_flatten=True, data_format='channels_last'):
    # Ensure the input shape is properly processed and validated.
    if input_shape is None:
        input_shape = (default_size, default_size, 3)
    
    # Check if input shape is in correct format
    if len(input_shape) == 3 and input_shape[-1] == 3:
        return input_shape
    else:
        raise ValueError("Input shape should be of the form (height, width, channels)")

def ShuffleNetV2(include_top=True,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='max',
                 input_shape=(224, 224, 3),
                 load_model=None,
                 num_shuffle_units=[3, 7, 3],
                 bottleneck_ratio=1,
                 classes=1000):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')

    # Model name including scale_factor, bottleneck_ratio, and num_shuffle_units
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    
    # Obtain input shape in a format suitable for the backend (replacing _obtain_input_shape with get_input_shape)
    input_shape = get_input_shape(input_shape, default_size=224, min_size=28, require_flatten=include_top,
                                  data_format=K.image_data_format())
    
    out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}

    # Check if pooling method is valid
    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')

    # Ensure scale_factor is a valid value
    if not (float(scale_factor) * 4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be a multiple of 4')

    # Calculate output channels for each stage
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  # Calculate output channels for each stage
    out_channels_in_stage[0] = 24  # First stage always has 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    # Handle input tensor for the model
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


            
    # # Get model inputs (whether from the input tensor or the specified img_input)
    # if input_tensor:
    #     inputs = get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input
    inputs = img_input

    # Create ShuffleNetV2 architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # Create stages with ShuffleNet units
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                  repeat=repeat,
                  bottleneck_ratio=bottleneck_ratio,
                  stage=stage + 2)

    # Final Conv2D layer before global pooling
    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    # Apply global pooling
    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    # Include top (final Dense layer with softmax activation)
    if include_top:
        x = Dense(classes, name='fc')(x)
        x = Activation('softmax', name='softmax')(x)


    # Create model
    model = Model(inputs, x, name=name)

    # Load pre-trained weights (if provided)
    if load_model:
        # Ensure that loading weights from a file works properly
        model.load_weights(load_model, by_name=True, skip_mismatch=True)

    return model

# Example usage
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU for testing purposes
    model = ShuffleNetV2(include_top=True, input_shape=(224, 224, 3), bottleneck_ratio=1)

    # Save the model architecture to a PNG file (optional, to visualize)
    plot_model(model, to_file='shufflenetv2.png', show_layer_names=True, show_shapes=True)
