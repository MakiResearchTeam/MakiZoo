from makiflow.layers import *
from makiflow.base import MakiTensor
import tensorflow as tf


def repeat_n_convLayers(
    x : MakiTensor,
    num_block : int,
    n : int,
    in_f=None,
    out_f=None,
    pooling_type='max_pool',
    use_bias=False,
    activation=tf.nn.relu,
    bn_params={},
    pool_params={}):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    num_block : int
        Number of block (used in name of layers).
    n : int
        Number of convolutions in blocks.
    pooling_type : str
        What type of pooling are will be used.
        'max_pool' - for max pooling.
        'avg_pool' - for average pooling.
        'none' or any other strings - the operation pooling will not be used.
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor).
    activation : tensorflow function
        The function of activation, by default tf.nn.relu.
    use_bias : bool
        Use bias on layers or not.
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor.
    """

    prefix_name = f'conv{num_block}/conv{num_block}_'

    if in_f is None:
        in_f = x.get_shape()[-1]

    if out_f is None:
        out_f = in_f * 2

    x = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=out_f,use_bias=use_bias,
                  activation=None,name=prefix_name + str(1))(x)
    x = ActivationLayer(activation=activation, name=prefix_name + 'activation_' + str(1))(x)

    for i in range(2,n+1):
        x = ConvLayer(kw=3,kh=3,in_f=out_f,out_f=out_f,use_bias=use_bias,
                      activation=None,name=prefix_name + str(i))(x)
        x = ActivationLayer(activation=activation, name=prefix_name + 'activation_' + str(i))(x)

    if pooling_type == 'max_pool':
        x = MaxPoolLayer(name=f'block{num_block}_pool', **pool_params)(x)
    elif pooling_type == 'avg_pool':
        x = AvgPoolLayer(name=f'block{num_block}_pool', **pool_params)(x)
    return x



