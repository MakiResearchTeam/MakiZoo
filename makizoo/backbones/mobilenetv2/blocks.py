# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiZoo.
#
# MakiZoo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiZoo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.


import numpy as np
import tensorflow as tf
from .utils import make_divisible
from makiflow.layers import (ConvLayer, BatchNormLayer,
                             ActivationLayer, DepthWiseConvLayer,
                             SumLayer)

NAME_EXPANDED_CONV = "expanded_conv"
ZERO_EXPANDED_CONV = NAME_EXPANDED_CONV + "/"
PREFIX = NAME_EXPANDED_CONV + "_{}/"

# Expand names
NAME_EXPAND = "{}expand/weights"
NAME_EXPAND_BN = "{}expand/BatchNorm"
NAME_EXPAND_ACT = "{}expand_relu"

# Depthwise names
NAME_DEPTHWISE = "{}depthwise/depthwise_weights"
NAME_DEPTHWISE_BN = "{}depthwise/BatchNorm"
NAME_DEPTHWISE_ACT = "{}depthsiwe_relu"

# Pointwise names
NAME_POINTWISE = "{}project/weights"
NAME_POINTWISE_BN = "{}project/BatchNorm"
NAME_FINAL_ADD = "{}add"


def inverted_res_block(
        x,
        expansion: int,
        alpha: float,
        block_id,
        out_f=None,
        in_f=None,
        stride=1,
        use_skip_connection=True,
        use_expand=True,
        activation=tf.nn.relu6,
        use_bias=False,
        bn_params={}):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    expansion : int
        Magnification multiplier of feature maps.
    alpha : float
        Controls the width of the network. This is known as the width multiplier in the MobileNetV2 paper.
        If alpha < 1.0, proportionally decreases the number of filters.
        If alpha > 1.0, proportionally increases the number of filters.
        If alpha = 1, default number of filters from the paper are used at each layer.
    block_id : int
        Number of block (used in name of layers).
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    out_f : int
        Number of output feature maps. By default None (shape will same as `in_f`).
    activation : tensorflow function
        The function of activation, by default tf.nn.relu6.
    use_bias : bool
        Use bias on layers or not.
    use_skip_connection : bool
        If true, sum input and output (if they are equal).
    use_expand : bool
        If true, input feature maps `in_f` will be expand to `expansion` * `in_f`.
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor

    """
    inputs = x

    if in_f is None:
        in_f = x.get_shape()[-1]

    # Calculate output number of f. for last ConvLayer, this number should be divisible by 8
    pointwise_f = make_divisible(int(out_f*alpha))

    prefix = PREFIX.format(str(block_id))

    # Standart cheme: expand -> depthwise -> pointwise
    if use_expand:
        # Expand stage, expand input f according to `expansion` value
        x = ConvLayer(
            kw=1,
            kh=1,
            in_f=in_f,
            out_f=int(expansion * in_f),
            name=NAME_EXPAND.format(prefix),
            use_bias=use_bias,
            activation=None,
        )(x)

        x = BatchNormLayer(D=x.get_shape()[-1], name=NAME_EXPAND_BN.format(prefix), **bn_params)(x)

        x = ActivationLayer(activation=activation, name=NAME_EXPAND_ACT.format(prefix))(x)
    else:
        # Expand layer is not used in first block
        # TODO: Add unique name for this layer, if we build some custom stuff
        prefix = ZERO_EXPANDED_CONV

    # Depthwise stage
    x = DepthWiseConvLayer(
        kw=3,
        kh=3,
        in_f=x.get_shape()[-1],
        multiplier = 1,
        activation=None,
        stride=stride,
        use_bias=use_bias,
        name=NAME_DEPTHWISE.format(prefix),
    )(x)

    x = BatchNormLayer(D=x.get_shape()[-1], name=NAME_DEPTHWISE_BN.format(prefix), **bn_params)(x)
    x = ActivationLayer(activation=activation, name=NAME_DEPTHWISE_ACT.format(prefix))(x)

    # Pointwise (Project) to certain size (input number of the f)
    x = ConvLayer(
        kw=1,
        kh=1,
        in_f=x.get_shape()[-1],
        out_f=pointwise_f,
        use_bias=use_bias,
        activation=None,
        name=NAME_POINTWISE.format(prefix)
    )(x)

    x = BatchNormLayer(D=x.get_shape()[-1], name=NAME_POINTWISE_BN.format(prefix), **bn_params)(x)

    if use_skip_connection or stride == 2:
        if x.get_shape()[-1] != inputs.get_shape()[-1]:
            raise ValueError(f'Error SumLayer\nIn block {block_id} input and output f. have different size')

        return SumLayer(name=NAME_FINAL_ADD.format(prefix))([inputs,x])
    else:
        return x

