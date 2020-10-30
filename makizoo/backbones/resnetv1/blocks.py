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


from makiflow.layers import *
from makiflow.core import MakiTensor
import tensorflow as tf
from .utils import get_batchnorm_params


PREFIX_NAME_BLOCK = "block{}/unit_{}"
PREFIX_NAME_SHORTCUT = "{}/bottleneck_v1/shortcut/{}"
PREFIX_NAME_LAYER = "{}/bottleneck_v1/conv{}/{}"

BATCH_NORM = "BatchNorm"
ACTIV = "activ"
WEIGHTS = "weights"
SUM_OPERATION = '/sum_operation'


def identity_block(
        x : MakiTensor,
        block_id: int,
        unit_id: int,
        num_block=None,
        in_f=None,
        use_bias=False,
        activation=tf.nn.relu,
        bn_params=None):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    block_id : int
        Number of block (used in name of layers).
    unit_id : int
        Unit of block (used in name of layers).
    num_block : int
        Number of sum operation (used in name of layers).
    bn_params : dict
        Parameters for BatchNormLayer. If equal to None all parameters will have default valued.

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor.
    """
    if bn_params is None:
        bn_params = get_batchnorm_params

    prefix_name = PREFIX_NAME_BLOCK.format(block_id, unit_id)
    if num_block is None:
        num_block = prefix_name + SUM_OPERATION
    else:
        num_block = str(num_block) + SUM_OPERATION

    if in_f is None:
        in_f = x.get_shape()[-1]

    reduction = int(in_f / 4)

    mx = ConvLayer(
        kw=1, kh=1, in_f=in_f, out_f=reduction, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 1, WEIGHTS)
    )(x)
    mx = BatchNormLayer(D=reduction, name=PREFIX_NAME_LAYER.format(prefix_name, 1, BATCH_NORM), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=PREFIX_NAME_LAYER.format(prefix_name, 1, ACTIV))(mx)

    mx = ConvLayer(
        kw=3, kh=3, in_f=reduction, out_f=reduction, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 2, WEIGHTS)
    )(mx)
    mx = BatchNormLayer(D=reduction, name=PREFIX_NAME_LAYER.format(prefix_name, 2, BATCH_NORM), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=PREFIX_NAME_LAYER.format(prefix_name, 2, ACTIV))(mx)

    mx = ConvLayer(
        kw=1, kh=1, in_f=reduction, out_f=in_f, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 3, WEIGHTS)
    )(mx)
    mx = BatchNormLayer(D=in_f, name=PREFIX_NAME_LAYER.format(prefix_name, 3, BATCH_NORM), **bn_params)(mx)

    x = SumLayer(name=num_block)([mx,x])

    return x


def conv_block(
        x : MakiTensor,
        block_id: int,
        unit_id: int,
        num_block=None,
        in_f=None,
        use_bias=False,
        activation=tf.nn.relu,
        stride=2,
        out_f=None,
        reduction=None,
        bn_params=None):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    out_f : int
        Output number of feature maps.
    block_id : int
        Number of block (used in name of layers).
    unit_id : int
        Unit of block (used in name of layers).
    num_block : int
        Number of sum operation (used in name of layers).
    bn_params : dict
        Parameters for BatchNormLayer. If equal to None all parameters will have default valued.

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor.
    """
    if bn_params is None:
        bn_params = get_batchnorm_params

    prefix_name = PREFIX_NAME_BLOCK.format(block_id, unit_id)

    if num_block is None:
        num_block = prefix_name + SUM_OPERATION
    else:
        num_block = str(num_block) + SUM_OPERATION

    if in_f is None:
        in_f = x.get_shape()[-1]

    if reduction is None:
        reduction = int(in_f / 2)

    if out_f is None:
        out_f = in_f * 2

    mx = ConvLayer(
        kw=1, kh=1, in_f=in_f, out_f=reduction, stride=stride, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 1, WEIGHTS)
    )(x)
    mx = BatchNormLayer(D=reduction, name=PREFIX_NAME_LAYER.format(prefix_name, 1, BATCH_NORM), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=PREFIX_NAME_LAYER.format(prefix_name, 1, ACTIV))(mx)

    mx = ConvLayer(
        kw=3, kh=3, in_f=reduction, out_f=reduction, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 2, WEIGHTS)
    )(mx)
    mx = BatchNormLayer(D=reduction, name=PREFIX_NAME_LAYER.format(prefix_name, 2, BATCH_NORM), **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=PREFIX_NAME_LAYER.format(prefix_name, 2, ACTIV))(mx)

    mx = ConvLayer(
        kw=1, kh=1, in_f=reduction, out_f=out_f, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_LAYER.format(prefix_name, 3, WEIGHTS)
    )(mx)
    mx = BatchNormLayer(D=out_f, name=PREFIX_NAME_LAYER.format(prefix_name, 3, BATCH_NORM), **bn_params)(mx)

    sx = ConvLayer(
        kw=1, kh=1, in_f=in_f, out_f=out_f, stride=stride, activation=None,
        use_bias=use_bias, name=PREFIX_NAME_SHORTCUT.format(prefix_name, WEIGHTS)
    )(x)
    sx = BatchNormLayer(D=out_f, name=PREFIX_NAME_SHORTCUT.format(prefix_name, BATCH_NORM), **bn_params)(sx)

    x = SumLayer(name=num_block)([mx,sx])

    return x


def without_pointwise_IB(
        x : MakiTensor,
        block_id: int,
        unit_id: int,
        num_block=None,
        in_f=None,
        use_bias=False,
        activation=tf.nn.relu,
        bn_params={}):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    block_id : int
        Number of block (used in name of layers).
    unit_id : int
        Unit of block (used in name of layers).
    num_block : int
        Number of sum operation (used in name of layers).
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default values.

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor.
    """

    prefix_name = 'stage' + str(block_id) + '_unit' + str(unit_id) + '_'

    if num_block is None:
        num_block = prefix_name + '/sum_operation'
    else:
        num_block = 'add_' + str(num_block)

    if in_f is None:
        in_f = x.get_shape()[-1]

    mx = BatchNormLayer(D=in_f, name=prefix_name + 'bn1', **bn_params)(x)
                                        
    mx = ActivationLayer(activation=activation, name=prefix_name + 'activation_1')(mx)

    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=prefix_name + 'zero_pad_1')(mx)

    mx = ConvLayer(kw=3, kh=3, in_f=in_f, out_f=in_f, activation=None,
                                padding='VALID', use_bias=use_bias, name=prefix_name + 'conv1')(mx)
                                                                                    
    mx = BatchNormLayer(D=in_f, name=prefix_name + 'bn2', **bn_params)(mx)

    mx = ActivationLayer(activation=activation, name=prefix_name + 'activation_2')(mx)

    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=prefix_name + 'zero_pad_2')(mx)

    mx = ConvLayer(kw=3, kh=3, in_f=in_f, out_f=in_f, activation=None,
                                padding='VALID', use_bias=use_bias, name=prefix_name + 'conv2')(mx)                        

    x = SumLayer(name=num_block)([mx,x])

    return x


def without_pointwise_CB(
        x : MakiTensor,
        block_id: int,
        unit_id: int,
        num_block=None,
        in_f=None,
        use_bias=False,
        activation=tf.nn.relu,
        stride=2,
        out_f=None,
        bn_params={}):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    in_f : int
        Number of input feature maps. By default is None (shape will be getted from tensor).
    out_f : int
        Number of output feature maps. By default is None which means out_f = 2 * in_f.
    activation : tensorflow function
        The function of activation. By default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    block_id : int
        Number of block (used in name of layers).
    unit_id : int
        Unit of block (used in name of layers).
    num_block : int
        Number of sum operation (used in name of layers).
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor.
    """
    prefix_name = 'stage' + str(block_id) + '_unit' + str(unit_id) + '_'

    if num_block is None:
        num_block = prefix_name + '/sum_operation'
    else:
        num_block = 'add_' + str(num_block)

    if in_f is None:
        in_f = x.get_shape()[-1]

    if out_f is None:
        out_f = int(2*in_f)

    x = BatchNormLayer(D=in_f, name=prefix_name + 'bn1', **bn_params)(x)
    x = ActivationLayer(activation=activation, name=prefix_name + 'activation_1')(x)

    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=prefix_name + 'zero_pad_1')(x)

    mx = ConvLayer(kw=3, kh=3, in_f=in_f, out_f=out_f, activation=None, stride=stride,
                                    padding='VALID', use_bias=use_bias, name=prefix_name + 'conv1')(mx)
                                                                                
    mx = BatchNormLayer(D=out_f, name=prefix_name + 'bn2', **bn_params)(mx)
    mx = ActivationLayer(activation=activation, name=prefix_name + 'activation_2')(mx)

    mx = ZeroPaddingLayer(padding=[[1,1],[1,1]], name=prefix_name + 'zero_pad_2')(mx)
    mx = ConvLayer(kw=3, kh=3, in_f=out_f, out_f=out_f, activation=None,
                                    padding='VALID', use_bias=use_bias, name=prefix_name + 'conv2')(mx)
                                                                                

    sx = ConvLayer(kw=1, kh=1, in_f=in_f, out_f=out_f, stride=stride,
                                    padding='VALID', activation=None, use_bias=use_bias, name=prefix_name + 'sc/conv')(x)
                                                                               
    x = SumLayer(name=num_block)([mx,sx])

    return x
