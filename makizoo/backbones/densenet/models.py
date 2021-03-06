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


import tensorflow as tf
from .builder import build_DenseNet


def DenseNet121(input_shape, classes=1000, include_top=False, create_model=False):
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
        bn_params={}
    )


def DenseNet161(input_shape, classes=1000, include_top=False, create_model=False):
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
        bn_params={}
    )


def DenseNet169(input_shape, classes=1000, include_top=False, create_model=False):
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
        bn_params={}
    )


def DenseNet201(input_shape, classes=1000, include_top=False, create_model=False):
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
        bn_params={}
    )


def DenseNet264(input_shape, classes=1000, include_top=False, create_model=False):
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
        bn_params={}
    )

