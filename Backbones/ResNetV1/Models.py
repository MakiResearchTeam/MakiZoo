from .Builder import build_resnet
import tensorflow as tf


def ResNet18(input_shape, classes=1000, include_top=False, create_model=True):

    if create_model:

        model = build_resnet(
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
        x, output = build_resnet(
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
        model = build_resnet(
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
        x, output = build_resnet(
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
        model = build_resnet(
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

        return model
    else:
        x, output = build_resnet(
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
        model = build_resnet(
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
        x, output = build_resnet(
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
        model = build_resnet(
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
        x, output = build_resnet(
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

