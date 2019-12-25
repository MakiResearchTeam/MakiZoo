import tensorflow as tf
from .builder import build_DenseNet


def DenseNet121(input_shape, classes=1000, include_top=False, create_model=False):
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 24, 16],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='DenseNet121',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        bm_params={}
    )


def DenseNet161(input_shape, classes=1000, include_top=False, create_model=False):
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 36, 24],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='DenseNet161',
        growth_rate=24,
        reduction=0.5,
        dropout_p_keep=0.8,
        bm_params={}
    )


def DenseNet169(input_shape, classes=1000, include_top=False, create_model=False):
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 32, 32],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='DenseNet169',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        bm_params={}
    )


def DenseNet201(input_shape, classes=1000, include_top=False, create_model=False):
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 48, 32],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='DenseNet201',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        bm_params={}
    )


def DenseNet264(input_shape, classes=1000, include_top=False, create_model=False):
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 64, 48],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='DenseNet264',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        bm_params={}
    )

