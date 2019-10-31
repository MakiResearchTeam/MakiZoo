from .Builder import build_ResNetV1, create_LittleResNetV1
import tensorflow as tf


# --------------------------------------------------------------------------------
#   Standard Residual Models V1
# --------------------------------------------------------------------------------

def ResNet18(input_shape, classes=1000, include_top=False, create_model=True):

    if create_model:

        model = build_ResNetV1(
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
        return model
    else:
        x, output = build_ResNetV1(
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

        return x, output


def ResNet34(input_shape, classes=1000, include_top=False, factorization_first_layer=False, create_model=True):

    if create_model:
        model = build_ResNetV1(
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

        return model
    else:
        x, output = build_ResNetV1(
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

        return x, output


def ResNet50(input_shape, classes=1000, include_top=False, factorization_first_layer=False, create_model=True):

    if create_model:
        model = build_ResNetV1(
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

        return model
    else:
        x, output = build_ResNetV1(
                    input_shape=input_shape,
                    repetition=(3, 4, 6, 3),
                    include_top=include_top,
                    num_classes=classes,
                    factorization_first_layer=factorization_first_layer,
                    use_bias=False,
                    activation=tf.nn.relu,
                    block_type='with_pointwise',
                    create_model=create_model,
                    name_model='ResNet50'
        )

        return x, output


def ResNet101(input_shape, classes=1000, include_top=False, factorization_first_layer=False, create_model=True):

    if create_model:
        model = build_ResNetV1(
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

        return model
    else:
        x, output = build_ResNetV1(
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

        return x, output


def ResNet152(input_shape, classes=1000, include_top=False, factorization_first_layer=False, create_model=True):

    if create_model:
        model = build_ResNetV1(
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

        return model
    else:
        x, output = build_ResNetV1(
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

        return x, output

# --------------------------------------------------------------------------------
#   Little version of the Residual Models V1
# --------------------------------------------------------------------------------
# Implementation taken from https://keras.io/examples/cifar10_resnet/

# Model parameter
# ------------------------------------------------------------\
#                  |      | Orig Paper |           | sec/epoch |
# Model            |layers| ResNet v1  |  Params   | GTX1080Ti |
#                  |  v1  | %Accuracy  |           | v1        |
# -------------------------------------------------------------|
# Little_ResNet20  | 20   | 91.25      |   0.27M   |    35     |
# Little_ResNet32  | 32   | 92.49      |   0.46M   |    50     |
# Little_ResNet44  | 44   | 92.83      |   0.66M   |    70     |
# Little_ResNet56  | 56   | 93.03      |   0.85M   |    90     |
# Little_ResNet110 | 110  | 93.39+-0.16|   1.7M    |    165    |
#-------------------------------------------------------------/

def Little_ResNet20(input_shape, classes=1000, include_top=False, create_model=True):

    if create_model:

        model = create_LittleResNetV1(
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
        return model
    else:
        x, output = create_LittleResNetV1(
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

        return x, output


def Little_ResNet32(input_shape, classes=1000, include_top=False, create_model=True):

    if create_model:

        model = create_LittleResNetV1(
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
        return model
    else:
        x, output = create_LittleResNetV1(
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

        return x, output


def Little_ResNet44(input_shape, classes=1000, include_top=False, create_model=True):

    if create_model:

        model = create_LittleResNetV1(
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
        return model
    else:
        x, output = create_LittleResNetV1(
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

        return x, output


def Little_ResNet56(input_shape, classes=1000, include_top=False, create_model=True):

    if create_model:

        model = create_LittleResNetV1(
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
        return model
    else:
        x, output = create_LittleResNetV1(
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

        return x, output


def Little_ResNet110(input_shape, classes=1000, include_top=False, create_model=True):

    if create_model:

        model = create_LittleResNetV1(
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
        return model
    else:
        x, output = create_LittleResNetV1(
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

        return x, output