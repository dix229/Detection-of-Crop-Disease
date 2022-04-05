import keras
import tensorflow as tf
from tensorflow.keras import optimizers
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPool1D,GlobalMaxPool2D,BatchNormalization,Activation
from keras.layers import Dense, Flatten,Dropout
from keras.models import Sequential,Model
from keras import regularizers
from keras import backend as K

from attention import cbam

adam = tf.keras.optimizers.Adam(lr=0.0005)
sgd = tf.keras.optimizers.SGD(lr=0.01)
weight_decay = 0.001
def LeNet5(input):

    conv1=Conv2D(32, kernel_size=(5, 5), activation='relu',padding='same')(input)
    at1 = cbam(conv1)
    maxpool1=MaxPooling2D(pool_size=(2,2))(at1)

    # dropout1=Dropout(0.2)(maxpool1)

    conv2=Conv2D(64, kernel_size=(5, 5), activation='relu',padding='same')(maxpool1)
    at2 = cbam(conv2)
    maxpool2=MaxPooling2D(pool_size=(2,2))(at2)


    flatten=Flatten()(maxpool2)
    dense1=Dense(500,activation='relu')(flatten)
    output=Dense(61,activation='softmax')(dense1)

    model = Model(input,output,name='lenet5-cbam')
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
#                kernel_regularizer=regularizers.l2(weight_decay),
#                bias_regularizer=regularizers.l2(weight_decay),
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
#         x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def AlexNet(input):
    # 第一层：//level one:
    conv1 = conv2d_bn(x=input, filters=96, kernel_size=[9, 9], strides=4,
                      padding='valid')  # input:224*224, output:54*54
    pooling1 = MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding='valid')(conv1)  # output:27*27

    # 第二层：//Second floor:
    conv2 = conv2d_bn(x=pooling1, filters=128, kernel_size=[5, 5], strides=1,
                      padding='same')  # output:27*27
    pooling2 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(conv2)  # output:13*13

    # 第三层//the third floor
    conv3 = conv2d_bn(x=pooling2, filters=192, kernel_size=[3, 3], strides=1,
                      padding='same')  # output:13*13

    # 第四层//Fourth floor
    conv4 = conv2d_bn(x=conv3, filters=192, kernel_size=[3, 3], strides=1,
                      padding='same')  # output:13*13
    pooling3 = MaxPooling2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(conv4)  # output:6*6

    # 第五层//Fifth floor
    conv5 = conv2d_bn(x=pooling3, filters=128, kernel_size=[3, 3], strides=1,
                      padding='same')  # output:6*6

    # 分类层//Classification layer
    pooling4 = GlobalAveragePooling2D()(conv5)
#     pooling4 = Flatten()(pooling4)
    output = Dense(61, activation='softmax', use_bias=True)(pooling4)

    model = keras.Model(input, output, name='alexnet')
    # 编译模型//Compile the model
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model
