import tensorflow as tf
import numpy as np
from makiflow.layers import *


def shuffle_net_basic_unit(x, out_f, stage, shuffle_group=2):
    # split in half!
    x1, x2 = ChannelSplitLayer(num_or_size_splits=2, axis=3, name=stage + '_split')(x)
    # 1 branch
    x = ConvLayer(
        kw=1, kh=1, in_f=x1.get_shape()[-1], out_f=out_f // 2, kernel_initializer=init_conv,
        use_bias=False, activation=None, name=stage + '_conv1'
    )(x1)
    x = BatchNormLayer(D=x.get_shape()[-1], name=stage + f'bn_1')(x)
    x = ActivationLayer(activation=actv, name=stage + f'activation_1')(x)

    x = DepthWiseConvLayer(kw=3, kh=3, in_f=x.get_shape()[-1], multiplier=1, kernel_initializer=init_conv,
                           use_bias=False, activation=None, name=stage + f'conv_2')(x)
    x = BatchNormLayer(D=x.get_shape()[-1], name=stage + f'bn_2')(x)

    x = ConvLayer(
        kw=1, kh=1, in_f=x.get_shape()[-1], out_f=x.get_shape()[-1], kernel_initializer=init_conv,
        use_bias=False, activation=None, name=stage + 'conv3'
    )(x)
    x = BatchNormLayer(D=x.get_shape()[-1], name=stage + f'bn_3')(x)
    x = ActivationLayer(activation=actv, name=stage + f'activation_3')(x)
    # Endo, concate
    x = ConcatLayer(name=stage + '_concat_f')([x2, x])
    x = ChannelShuffleLayer(num_groups=shuffle_group, name=stage + '_shuffle_f')(x)
    return x


def shuffle_net_spatial_down_samp_unit(x, out_f, stage, shuffle_group=2, stride=2):
    # 1 branch
    fx = ConvLayer(
        kw=1, kh=1, in_f=x.get_shape()[-1], out_f=out_f // 2, kernel_initializer=init_conv,
        use_bias=False, activation=None, name=stage + 'fx_conv1'
    )(x)
    fx = BatchNormLayer(D=fx.get_shape()[-1], name=stage + f'fx_bn_1')(fx)
    fx = ActivationLayer(activation=actv, name=stage + f'fx_activation_1')(fx)

    fx = DepthWiseConvLayer(kw=3, kh=3, in_f=fx.get_shape()[-1], multiplier=1, kernel_initializer=init_conv,
                            use_bias=False, stride=stride, activation=None, name=stage + f'fx_conv_2')(fx)
    fx = BatchNormLayer(D=fx.get_shape()[-1], name=stage + f'fx_bn_2')(fx)

    fx = ConvLayer(
        kw=1, kh=1, in_f=fx.get_shape()[-1], out_f=fx.get_shape()[-1], kernel_initializer=init_conv,
        use_bias=False, activation=None, name=stage + 'fx_conv3'
    )(fx)
    fx = BatchNormLayer(D=fx.get_shape()[-1], name=stage + f'fx_bn_3')(fx)
    fx = ActivationLayer(activation=actv, name=stage + f'fx_activation_3')(fx)
    # 2 Branch
    sx = DepthWiseConvLayer(kw=3, kh=3, in_f=x.get_shape()[-1], multiplier=1, kernel_initializer=init_conv,
                            use_bias=False, stride=stride, activation=None, name=stage + f'sx_conv_1')(x)
    sx = BatchNormLayer(D=sx.get_shape()[-1], name=stage + f'sx_bn_1')(sx)

    sx = ConvLayer(
        kw=1, kh=1, in_f=x.get_shape()[-1], out_f=out_f // 2, kernel_initializer=init_conv,
        use_bias=False, activation=None, name=stage + 'sx_conv2'
    )(sx)
    sx = BatchNormLayer(D=sx.get_shape()[-1], name=stage + f'sx_bn_2')(sx)
    sx = ActivationLayer(activation=actv, name=stage + f'sx_activation_2')(sx)

    x = ConcatLayer(name=stage + '_concat_f')([fx, sx])
    x = ChannelShuffleLayer(num_groups=shuffle_group, name=stage + '_shuffle_f')(x)
    return x