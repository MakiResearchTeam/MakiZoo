import tensorflow as tf
from .builder import build_MobileNetV2

def MobileNetV2_1_0(input_shape, classes=1000, include_top=False, create_model=False):
    return  build_MobileNetV2(
        input_shape=input_shape,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='MobileNetV2_1_0',
        alpha=1.0,
        expansion=6
    )

def MobileNetV2_1_4(input_shape, classes=1000, include_top=False, create_model=False):
    return  build_MobileNetV2(
        input_shape=input_shape,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='MobileNetV2_1_4',
        alpha=1.4,
        expansion=6
    )

def MobileNetV2_0_75(input_shape, classes=1000, include_top=False, create_model=False):
    return  build_MobileNetV2(
        input_shape=input_shape,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='MobileNetV2_1_0',
        alpha=0.75,
        expansion=6
    )

def MobileNetV2_1_3(input_shape, classes=1000, include_top=False, create_model=False):
    return  build_MobileNetV2(
        input_shape=input_shape,
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        activation=tf.nn.relu6,
        create_model=create_model,
        name_model='MobileNetV2_1_0',
        alpha=1.3,
        expansion=6
    )
