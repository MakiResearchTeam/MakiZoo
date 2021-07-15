# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiZoo.
#
# MakiZoo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiZoo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from makiflow.layers.utils import InitConvKernel
import tensorflow as tf
from .builder import build_DenseNet


def DenseNet121(input_shape, classes=1000, include_top=False, create_model=False, kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet18 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used

    Returns
    -------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 24, 16],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet121',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        kernel_initializer=kernel_initializer,
        bn_params={}
    )


def DenseNet161(input_shape, classes=1000, include_top=False, create_model=False, kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet18 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used

    Returns
    -------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 36, 24],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet161',
        growth_rate=24,
        reduction=0.5,
        dropout_p_keep=0.8,
        kernel_initializer=kernel_initializer,
        bn_params={}
    )


def DenseNet169(input_shape, classes=1000, include_top=False, create_model=False, kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet18 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used

    Returns
    -------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 32, 32],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet169',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        kernel_initializer=kernel_initializer,
        bn_params={}
    )


def DenseNet201(input_shape, classes=1000, include_top=False, create_model=False, kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet18 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used

    Returns
    -------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 48, 32],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet201',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        kernel_initializer=kernel_initializer,
        bn_params={}
    )


def DenseNet264(input_shape, classes=1000, include_top=False, create_model=False, kernel_initializer=InitConvKernel.HE):
    """
    Create ResNet18 model with certain `input_shape`

    Parameters
    ----------
    input_shape : list
        Input shape into model,
        Example: [1, 300, 300, 3]
    classes : int
        Number of classes for classification task, used if `include_top` is True
    include_top : bool
        If equal to True then additional dense layers will be added to the model,
        In order to build full ResNet18 model
    create_model : bool
        If equal to True then will be created Classification model
        and this method wil return only this obj
    kernel_initializer : str
        Name of type initialization for conv layers,
        For more examples see: makiflow.layers.utils,
        By default He initialization are used

    Returns
    -------
    if `create_model` is False
        in_x : mf.MakiTensor
            Input MakiTensor
        output : mf.MakiTensor
            Output MakiTensor
    if `create_model` is True
        model : mf.models.Classificator
            Classification model

    """
    return build_DenseNet(
        input_shape=input_shape,
        nb_layers=[6, 12, 64, 48],
        include_top=include_top,
        num_classes=classes,
        use_bias=False,
        use_bottleneck=True,
        subsample_initial_block=True,
        activation=tf.nn.relu,
        create_model=create_model,
        name_model='DenseNet264',
        growth_rate=32,
        reduction=0.5,
        dropout_p_keep=0.8,
        kernel_initializer=kernel_initializer,
        bn_params={}
    )

