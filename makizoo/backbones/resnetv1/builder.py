from .blocks import identity_block as with_pointwise_IB
from .blocks import conv_block as with_pointwise_CB

from .blocks import without_pointwise_IB
from .blocks import without_pointwise_CB

from .utils import get_batchnorm_params

from makiflow.layers import *
from makiflow.models import Classificator
import tensorflow as tf


def build_ResNetV1(
    input_shape,
    repetition=(2,2,2,2),
    include_top=False,
    num_classes=1000,
    factorization_first_layer=False,
    use_bias=False,
    activation=tf.nn.relu,
    block_type='with_pointwise',
    create_model=False,
    name_model='MakiClassificator',
    init_filters=64,
    min_reduction=64,
    activation_between_blocks=True):
    """
    Parameters
    ----------
        input_shape : List
            Input shape of neural network. Example - [32, 128, 128, 3]
            which mean 32 - batch size, two 128 - size of picture, 3 - number of colors
        repetition : list
            Number of repetition on certain depth
        include_top : bool
            If true when at the end of the neural network added Global Avg pooling and Dense Layer wothout
            activation with the number of output neurons equal to num_classes
        factorization_first_layer : bool
            If true at the start of CNN factorizate convolution layer into 3 convolution layers
        use_bias : bool
            If true, when on layers used bias operation
        activation : tf object
            Activation on every convolution layer
        block_type : str
            Type of blocks.
            with_pointwise - use pointwise operation in blocks, usually used in ResNet50, ResNet101, ResNet152
            without_pointwise - block without pointwise operation, usually  used in ResNet18, ResNet34
        create_model : bool
            Return build classification model, otherwise return input MakiTensor and output MakiTensor
        name_model : str
            Name of model, if it will be created
        init_filters : int
            Started number of feature maps
        min_reduction : int
            Minimum reduction in blocks
        activation_between_blocks : bool
            Use activation between blocks
    Returns
    ---------
        x : MakiTensor
            Output MakiTensor
        out_f : int
            Output number of feature maps
    """

    if repetition is not list or len(repetition) != 4:
        raise TypeError('repetition should be list of size 4')


    feature_maps = init_filters
    bm_params = get_batchnorm_params()

    if block_type == 'with_pointwise':
        conv_block = with_pointwise_CB
        iden_block = with_pointwise_IB
        output_factorization_layer = init_filters
        pointwise = True
    elif block_type == 'without_pointwise':
        conv_block = without_pointwise_CB
        iden_block = without_pointwise_IB
        output_factorization_layer = init_filters * 2
        pointwise = False
    else:
        raise Exception(f'{block_type} type is not found')

    in_x = InputLayer(input_shape=input_shape,name='Input')

    if factorization_first_layer:

        x = ConvLayer(kw=3, kh=3, in_f=input_shape[-1], out_f=feature_maps, use_bias=use_bias,
                                    activation=None, name='conv1_1/weights')(in_x)

        x = BatchNormLayer(D=feature_maps, name='conv1_1/BatchNorm', **bm_params)(x)
        x = ActivationLayer(activation=activation, name='conv1_1/activation')(x)

        x = ConvLayer(kw=3, kh=3, in_f=feature_maps, out_f=feature_maps, use_bias=use_bias,
                                    activation=None, name='conv1_2/weights')(x)

        x = BatchNormLayer(D=feature_maps, name='conv1_2/BatchNorm', **bm_params)(x)
        x = ActivationLayer(activation=activation, name='conv1_2/activation')(x)

        x = ConvLayer(kw=3, kh=3, in_f=feature_maps, out_f=output_factorization_layer,
                                    use_bias=use_bias, stride=2, activation=None, name='conv1_3/weights')(x)

        x = BatchNormLayer(D=output_factorization_layer, name='conv1_3/BatchNorm', **bm_params)(x)
        x = ActivationLayer(activation=activation, name='conv1_3/activation')(x)

        feature_maps = output_factorization_layer
    else:
        x = ConvLayer(kw=7, kh=7, in_f=input_shape[-1], out_f=feature_maps, use_bias=use_bias,
                                    stride=2, activation=None,name='conv1/weights')(in_x)
        
        x = BatchNormLayer(D=feature_maps, name='conv1/BatchNorm', **bm_params)(x)
        x = ActivationLayer(activation=activation, name='activation')(x)
    
    x = MaxPoolLayer(ksize=[1,3,3,1], name='max_pooling2d')(x)

    # Build body of ResNet
    num_activation = 3
    num_block = 0

    for stage, repeat in enumerate(repetition):
        for block in range(repeat):

            # First block of the first stage is used without strides because we have maxpooling before
            if block == 0 and stage == 0:
                if pointwise:
                    x = conv_block(
                        x=x, 
                        block_id=stage, 
                        unit_id=block, 
                        num_block=num_block,
                        use_bias=use_bias,
                        activation=activation,
                        stride=1,
                        out_f=256,
                        reduction=min_reduction,
                        bm_params=bm_params
                    )[0]
                else:
                    x = conv_block(
                        x=x, 
                        block_id=stage, 
                        unit_id=block, 
                        num_block=num_block,
                        use_bias=use_bias,
                        activation=activation,
                        stride=1,
                        out_f=init_filters,
                        bm_params=bm_params
                    )[0]
            elif block == 0:
                # Every first block in new stage (zero block) we do block with stride 2 and increase number of feature maps
                x = conv_block(
                    x=x, 
                    block_id=stage, 
                    unit_id=block, 
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    stride=2,
                    bm_params=bm_params
                )[0]
            else:
                x = iden_block(
                    x=x,
                    block_id=stage,
                    unit_id=block,
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    bm_params=bm_params
                )[0]
            num_block += 1

            if activation_between_blocks:
                x = ActivationLayer(activation=activation, name='activation_' + str(num_activation))(x)
                num_activation += 3
    
    if not pointwise:
        x = BatchNormLayer(D=x.get_shape()[-1], name='bn1', **bm_params)(x)
        x = ActivationLayer(activation=activation, name='relu1')(x)

    if include_top:
        x = GlobalAvgPoolLayer(name='avg_pool')(x)
        output = DenseLayer(in_d=x.get_shape()[-1], out_d=num_classes, activation=None, name='logits')(x)
    else:
        output = x

    if create_model:
        return Classificator(in_x,output,name=name_model)
    else:
        return in_x, output


def create_LittleResNetV1(
        input_shape,
        depth=20,
        include_top=False,
        num_classes=1000,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=False,
        name_model='MakiClassificator',
        activation_between_blocks=True):
        
    feature_maps = 16
    bm_params = get_batchnorm_params()

    conv_block = without_pointwise_CB
    iden_block = without_pointwise_IB

    in_x = InputLayer(input_shape=input_shape,name='Input')

    x = ConvLayer(kw=3, kh=3, in_f=input_shape[-1], out_f=feature_maps, activation=None,
                                    use_bias=use_bias, name='conv1')(in_x)
                                                                                
    x = BatchNormLayer(D=feature_maps, name='bn_1', **bm_params)(x)
    x = ActivationLayer(activation=activation, name= 'activation_1')(x)

    repeat = int((depth - 2) / 6)

    # Build body of ResNet
    num_block = 0
    num_activation = 3
    
    for stage in range(3):
        for block in range(repeat):

            # First block of the first stage is used without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = conv_block(
                    x=x, 
                    block_id=stage, 
                    unit_id=block, 
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    stride=1,
                    out_f=feature_maps,
                    bm_params=bm_params
                )[0]
            elif block == 0:
                # Every first block in new stage (zero block) we do block with stride 2 and increase number of feature maps
                x = conv_block(
                    x=x, 
                    block_id=stage, 
                    unit_id=block, 
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    stride=2,
                    bm_params=bm_params
                )[0]
            else:
                x = iden_block(
                    x=x,
                    block_id=stage,
                    unit_id=block,
                    num_block=num_block,
                    use_bias=use_bias,
                    activation=activation,
                    bm_params=bm_params
                )[0]

            if activation_between_blocks:
                x = ActivationLayer(activation=activation, name='activation_' + str(num_activation))(x)
                num_activation += 3
            num_block += 1

    if include_top:
        x = GlobalAvgPoolLayer(name='avg_pool')(x)
        output = DenseLayer(in_d=x.get_shape()[-1], out_d=num_classes, activation=None, name='logits')(x)
    else:
        output = x

    if create_model:
        return Classificator(in_x,output,name=name_model)
    else:
        return in_x, output