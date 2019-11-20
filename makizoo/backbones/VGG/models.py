import tensorflow as tf
from .builder import build_VGG


def VGG16(input_shape, classes=1000, include_top=False, create_model=False):
    return build_VGG(
            input_shape=input_shape,
            repetition=3,
            include_top=include_top,
            num_classes=classes,
            use_bias=False,
            activation=tf.nn.relu,
            create_model=create_model,
            name_model='VGG16'
    )


def VGG19(input_shape, classes=1000, include_top=False, create_model=False):
    return build_VGG(
            input_shape=input_shape,
            repetition=4,
            include_top=include_top,
            num_classes=classes,
            use_bias=False,
            activation=tf.nn.relu,
            create_model=create_model,
            name_model='VGG19'
    )