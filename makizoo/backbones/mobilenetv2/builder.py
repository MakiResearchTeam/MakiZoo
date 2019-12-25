import tensorflow as tf
from .blocks import inverted_res_block
from .utils import make_divisible, get_batchnorm_params

from makiflow.layers import *
from makiflow.models import Classificator

def build_MobileNetV2(
        input_shape,
        include_top=False,
        num_classes=1000,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=False,
        name_model='MakiClassificator',
        alpha=1,
        expansion=6,
        bm_params={}):
    """
    Parameters
    ----------
    x : MakiTensor
        Input MakiTensor.
    expansion : int
        Magnification multiplier of feature maps.
    input_shape : List
        Input shape of neural network. Example - [32, 128, 128, 3]
        which mean 32 - batch size, two 128 - size of picture, 3 - number of colors.
    alpha : int
        Controls the width of the network. This is known as the width multiplier in the MobileNetV2 paper.
        If alpha < 1.0, proportionally decreases the number of filters.
        If alpha > 1.0, proportionally increases the number of filters.
        If alpha = 1, default number of filters from the paper are used at each layer.
    include_top : bool
        If true when at the end of the neural network added Global Avg pooling and Dense Layer without
        activation with the number of output neurons equal to num_classes.
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
    create_model : bool
        Return build classification model, otherwise return input MakiTensor and output MakiTensor.
    name_model : str
        Name of model, if it will be created.
    num_classes : int
        Number of classes that you need to classify.
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

    first_filt = make_divisible(32 * alpha, 8)

    in_x = InputLayer(input_shape=input_shape, name='input')

    x = ConvLayer(kw=3,
                    kh=3,
                    in_f=3,
                    out_f=first_filt,
                    stride=2,
                    padding='SAME',
                    activation=None,
                    use_bias=use_bias,
                    name='Conv/weights',
    )(in_x)

    x = BatchNormLayer(D=first_filt, name='Conv/BatchNorm', **bm_params)(x)
    x = ActivationLayer(activation=activation, name='Conv_relu')(x)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=16, alpha=alpha, expansion=1,
                                    block_id=0, use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=False, use_skip_connection=False)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=24, alpha=alpha, stride=2, expansion=expansion,
                                    block_id=1, use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=False)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=24, alpha=alpha, expansion=expansion, block_id=2,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=32, alpha=alpha, stride=2,expansion=expansion,
                                    block_id=3, use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=False)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=32, alpha=alpha,expansion=expansion,block_id=4,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=32, alpha=alpha, expansion=expansion, block_id=5,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=64, alpha=alpha, stride=2, expansion=expansion,
                                    block_id=6, use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=False)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1],out_f=64, alpha=alpha, expansion=expansion, block_id=7,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=64, alpha=alpha, expansion=expansion, block_id=8,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1],out_f=64, alpha=alpha, expansion=expansion, block_id=9,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=96, alpha=alpha,expansion=expansion, block_id=10,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=96, alpha=alpha,expansion=expansion, block_id=11,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=96, alpha=alpha,expansion=expansion, block_id=12,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=160, alpha=alpha, stride=2, expansion=expansion, block_id=13,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=False)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=160, alpha=alpha, expansion=expansion, block_id=14,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=160, alpha=alpha, expansion=expansion, block_id=15,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=True)

    x = inverted_res_block(inputs=x, in_f=x.get_shape()[-1], out_f=320, alpha=alpha, expansion=expansion, block_id=16,
                                    use_bias=use_bias, activation=activation,
                                    bm_params=bm_params, use_expand=True, use_skip_connection=False)

    if alpha > 1.0:
        last_block_filters = make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = ConvLayer(kh=1,
        kw=1,
        in_f=in_new_f,
        out_f=last_block_filters,
        stride=1,
        padding='SAME',
        activation=None,
        use_bias=use_bias,
        name='Conv_1/weights',
    )(x)

    x = BatchNormLayer(D=last_block_filters, name='Conv_1/BatchNorm', **bm_params)(x)
    pred_top = ActivationLayer(activation=activation, name='out_relu')(x)

    if include_top:
        x = GlobalAvgPoolLayer(name='global_avg')(pred_top)
        x = ReshapeLayer(new_shape=[batch_size,1,1,1280],name='resh')(x)
        x = ConvLayer(kw=1,kh=1,in_f=1280,out_f=num_classes,name='prediction')(x)
        output = ReshapeLayer(new_shape=[batch_size,num_classes],name='endo')(x)

        if create_model:
            return Classificator(in_x, output, name_model)
    else:
        output = pred_top

    return in_x, output

