from makiflow.layers import *
from makiflow.base import MakiTensor
import tensorflow as tf


def repeat_n_convLayers(
    x : MakiTensor,
    num_block : int,
    n : int,
    in_f=None,
    out_f=None,
    use_maxpoolLayer=False,
    use_bias=False,
    activation=tf.nn.relu,
    bn_params={},
    maxpool_params={}):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor
    num_block : int
        Number of block (used in name of layers)
    n : int
        Number of convolutions in blocks
    use_maxpoolLayer : bool
        If true, after all convolutions will be used MaxPoolLayer
    in_f : int
        Number of input feature maps. By default None (shape will be getted from tensor)
    activation : tensorflow function
        The function of activation, by default tf.nn.relu
    use_bias : bool
        Use bias on layers or not
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued

    Returns
    ---------
    x : MakiTensor
        Output MakiTensor
    in_f : int
        Output number of feature maps
    """

    prefix_name = f'conv{num_block}/conv{num_block}_'

    if in_f is None:
        in_f = x.get_shape()[-1]

    if out_f is None:
        out_f = in_f * 2

    x = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=out_f,use_bias=use_bias,
                  activation=activation,name=prefix_name + str(1))(x)

    for i in range(2,n+1):
        x = ConvLayer(kw=3,kh=3,in_f=out_f,out_f=out_f,use_bias=use_bias,
                      activation=activation,name=prefix_name + str(i))(x)

    if use_maxpoolLayer:
        x = MaxPoolLayer(name=f'block{num_block}_pool', **maxpool_params)(x)

    return x, out_f



