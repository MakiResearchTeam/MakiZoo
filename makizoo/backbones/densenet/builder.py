import tensorflow as tf
from .blocks import transition_layer, bottleneck_layer, dense_block
from .utils import get_batchnorm_params

from makiflow.layers import *
from makiflow.models import Classificator

def build_DenseNet(
        input_shape,
        nb_layers=[6,12,24,16],
        include_top=False,
        num_classes=1000,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=False,
        name_model='MakiClassificator',
        growth_rate=32,
        reduction=0.0,
        dropout_p_keep=None,
        bm_params={}
    ):
    """
     Parameters
     ----------
     x : MakiTensor
         Input MakiTensor.
     nb_layers : int
         List of lenght 4, where `nb_layers[i]` is number of repetition layers at stage `i` (i from 0 to 3).
    growth_rate : int
        Coefficient `k` from original papep, https://arxiv.org/pdf/1608.06993.pdf .
    dropout_p_keep : float
        The probability that each element of x is not discarded.
    reduction : float
        Coefficient, where `r` = 1 - `rediction`, `r` is how much number of feature maps need to compress in transition layers.
    input_shape : List
        Input shape of neural network. Example - [32, 128, 128, 3]
        which mean 32 - batch size, two 128 - size of picture, 3 - number of colors.
    create_model : bool
        Return build classification model, otherwise return input MakiTensor and output MakiTensor.
    name_model : str
        Name of model, if it will be created.
    num_classes : int
        Number of classes that you need to classify.
    bn_params : dict
        Parameters for BatchNormLayer. If empty all parameters will have default valued.
    include_top : bool
        If true when at the end of the neural network added Global Avg pooling and Dense Layer without
        activation with the number of output neurons equal to num_classes.
    activation : tensorflow function
        The function of activation, by default tf.nn.relu6.

     Returns
     ---------
     in_x : MakiTensor
         Input MakiTensor.
     output : int
         Output MakiTensor.
     Classificator : MakiFlow.Classificator
         Constructed model.
     """
    if bm_params is None or len(bm_params) == 0:
        bm_params = get_batchnorm_params()
    compression = 1 - reduction

    in_x = InputLayer(input_shape=[batch_size,picture_size,picture_size,3], name='Input')

    x = ZeroPaddingLayer(padding=[[3,3],[3,3]], name='zero_padding2d_4')(in_x)

    x = ConvLayer(kw=7,kh=7,in_f=3, stride=2, out_f=growth_rate * 2, activation=None, use_bias=use_bias,
            name='conv1/conv', padding='VALID')(x)

    x = BatchNormLayer(D=growth_rate * 2, name='conv1/bn', **bm_params)(x)
    x = ActivationLayer(activation=activation, name='conv1/relu')(x)
    x = ZeroPaddingLayer(padding=[[1,1],[1,1]], name='zero_padding2d_5')(x)

    x = MaxPoolLayer(ksize=[1,3,3,1], padding='VALID', name='pool1')(x)

    # densenet blocks
    for block_index in range(len(nb_layers) - 1):
        # dense block
        x = dense_block(x=x, nb_layers=nb_layers[block_index], stage=block_index+2,
                growth_rate=growth_rate, dropout_p_keep=dropout_p_keep,
                        activation=activation, use_bias=use_bias, bm_params=bm_params)

        # transition block
        x = transition_layer(x=x,
             dropout_p_keep=dropout_p_keep, number=block_index+2, compression=compression,
                        activation=activation, use_bias=use_bias, bm_params=bm_params)

    x = dense_block(x=x, nb_layers=nb_layers[-1], stage=len(nb_layers)+1,
            growth_rate=growth_rate, dropout_p_keep=dropout_p_keep,
                        activation=activation, use_bias=use_bias, bm_params=bm_params)

    x = BatchNormLayer(D=x.get_shape()[-1], name='bn', **bm_params)(x)
    x = ActivationLayer(activation=activation, name='relu')(x)
    if include_top:
        x = GlobalAvgPoolLayer(name='avg_pool')(x)
        # dense part (fc layers)
        output = DenseLayer(in_d=x.get_shape()[-1], out_d=num_classes, activation=None, use_bias=True, name="fc1000")(x)
        if create_model:
            return Classificator(in_x, output, name_model)
    else:
        output = x

    return in_x, output

