from .builder import build_ResNetV1, build_LittleResNetV1
import tensorflow as tf


# --------------------------------------------------------------------------------
#   Standard Residual Models V1
# --------------------------------------------------------------------------------

def ResNet18(input_shape, classes=1000, include_top=False, create_model=False):
    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(2, 2, 2, 2),
        include_top=include_top,
        num_classes=classes,
        factorization_first_layer=False,
        use_bias=False,
        activation=tf.nn.relu,
        block_type='without_pointwise',
        create_model=create_model,
        name_model='ResNet18'
    )


def ResNet34(input_shape, classes=1000, include_top=False, factorization_first_layer=False, create_model=False):
    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(3, 4, 6, 3),
        include_top=include_top,
        num_classes=classes,
        factorization_first_layer=factorization_first_layer,
        use_bias=False,
        activation=tf.nn.relu,
        block_type='without_pointwise',
        create_model=create_model,
        name_model='ResNet34'
    )


def ResNet50(input_shape, classes=1000, include_top=False, factorization_first_layer=False, create_model=False):
    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(3, 4, 6, 3),
        include_top=include_top,
        num_classes=classes,
        factorization_first_layer=factorization_first_layer,
        use_bias=False,
        activation=tf.nn.relu,
        block_type='with_pointwise',
        create_model=create_model,
        name_model='ResNet50',
    )


def ResNet101(input_shape, classes=1000, include_top=False, factorization_first_layer=False, create_model=False):
    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(3, 4, 23, 3),
        include_top=include_top,
        num_classes=classes,
        factorization_first_layer=factorization_first_layer,
        use_bias=False,
        activation=tf.nn.relu,
        block_type='with_pointwise',
        create_model=create_model,
        name_model='ResNet101'
    )


def ResNet152(input_shape, classes=1000, include_top=False, factorization_first_layer=False, create_model=False):
    return build_ResNetV1(
        input_shape=input_shape,
        repetition=(3, 8, 36, 3),
        include_top=include_top,
        num_classes=classes,
        factorization_first_layer=factorization_first_layer,
        use_bias=False,
        activation=tf.nn.relu,
        block_type='with_pointwise',
        create_model=create_model,
        name_model='ResNet152'
    )


# --------------------------------------------------------------------------------
#   Little version of the Residual Models V1
# --------------------------------------------------------------------------------
# Implementation taken from https://keras.io/examples/cifar10_resnet/


def Little_ResNet20(input_shape, classes=1000, include_top=False, create_model=False):
    return build_LittleResNetV1(
        input_shape,
        depth=20,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet20',
        activation_between_blocks=True
    )


def Little_ResNet32(input_shape, classes=1000, include_top=False, create_model=False):
    return build_LittleResNetV1(
        input_shape,
        depth=32,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet32',
        activation_between_blocks=True
    )


def Little_ResNet44(input_shape, classes=1000, include_top=False, create_model=False):
    return build_LittleResNetV1(
        input_shape,
        depth=44,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet44',
        activation_between_blocks=True
    )


def Little_ResNet56(input_shape, classes=1000, include_top=False, create_model=False):
    return build_LittleResNetV1(
        input_shape,
        depth=56,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet56',
        activation_between_blocks=True
    )


def Little_ResNet110(input_shape, classes=1000, include_top=False, create_model=False):
    return build_LittleResNetV1(
        input_shape,
        depth=110,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='Little_ResNet110',
        activation_between_blocks=True
    )
