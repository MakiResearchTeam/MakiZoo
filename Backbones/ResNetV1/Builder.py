from .Blocks import identity_block as with_pointwise_IB
from .Blocks import conv_block as with_pointwise_CB

from .Blocks import without_pointwise_IB
from .Blocks import without_pointwise_CB

from makiflow.layers import *
from makiflow.models import Classificator
import tensorflow as tf


def build_resnet(
        input_shape,
        repetition=(2,2,2,2),
        include_top=False,
        num_classes=1000,
        factorization_first_layer=False,
        use_bias=False,
        activation=tf.nn.relu,
        block_type='with_pointwise',
        create_model=False,
        name_model='MakiClassificator'):

    feature_maps = 64

    if block_type == 'with_pointwise':
        conv_block = with_pointwise_CB
        iden_block = with_pointwise_IB
        output_factorization_layer = 64
        pointwise = True
    elif block_type == 'without_pointwise':
        conv_block = without_pointwise_CB
        iden_block = without_pointwise_IB
        output_factorization_layer = 128
        pointwise = False
    else:
        raise Exception(f'{block_type} type is not found')

    in_x = InputLayer(input_shape=input_shape,name='Input')

    if factorization_first_layer:
        x = ConvLayer(kw=3, kh=3, in_f=3, out_f=feature_maps, use_bias=use_bias,
                                    activation=None, name='conv1_1/weights')(in_x)

        x = BatchNormLayer(D=feature_maps, name='conv1_1/BatchNorm')(x)
        x = ActivationLayer(activation=activation, name='conv1_1/activation')(x)

        x = ConvLayer(kw=3, kh=3, in_f=feature_maps, out_f=feature_maps, use_bias=use_bias,
                                    activation=None, name='conv1_2/weights')(x)

        x = BatchNormLayer(D=feature_maps, name='conv1_2/BatchNorm')(x)
        x = ActivationLayer(activation=activation, name='conv1_2/activation')(x)

        x = ConvLayer(kw=3, kh=3, in_f=feature_maps, out_f=output_factorization_layer,
                                    use_bias=use_bias, stride=2, activation=None, name='conv1_3/weights')(x)

        x = BatchNormLayer(D=output_factorization_layer, name='conv1_3/BatchNorm')(x)
        x = ActivationLayer(activation=activation, name='conv1_3/activation')(x)

        feature_maps = output_factorization_layer
    else:
        x = ConvLayer(kw=7, kh=7, in_f=input_shape[-1], out_f=feature_maps, use_bias=use_bias,
                                    stride=2, activation=None,name='conv1/weights')(in_x)
        
        x = BatchNormLayer(D=feature_maps, name='conv1/BatchNorm')(x)
        x = ActivationLayer(activation=activation, name='activation')(x)

    x = MaxPoolLayer(ksize=[1,3,3,1], name='max_pooling2d')(x)

    # Build body of ResNet
    num_activation = 3

    for stage, repeat in enumerate(repetition):
        for block in range(repeat):

            # First block of the first stage is used without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = conv_block(
                    x=x, 
                    block_id=stage, 
                    unit_id=block, 
                    num_block=( stage + 1) * block,
                    use_bias=use_bias,
                    activation=activation,
                    stride=1
                )[0]
            # Every first block in new stage (zero block) we do block with stride 2 and increase number of feature maps
            elif block == 0:
                x = conv_block(
                    x=x, 
                    block_id=stage, 
                    unit_id=block, 
                    num_block=( stage + 1) * block,
                    use_bias=use_bias,
                    activation=activation,
                    stride=2
                )[0]

            else:
                x = iden_block(
                    x=x,
                    block_id=stage,
                    unit_id=block,
                    num_block=( stage + 1) * block,
                    use_bias=use_bias,
                    activation=activation
                )[0]
            
            if pointwise:
                x = ActivationLayer(activation=activation, name='activation_' + str(num_activation))(x)
                num_activation += 3
    
    if not pointwise:
        x = BatchNormLayer(D=x.get_shape()[-1], name='bn1')(x)
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