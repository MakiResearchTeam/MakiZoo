# [(out_channel, repeat_times), (out_channel, repeat_times), ...]
size2feat = {
    "0.5c": [(48, 4), (96, 6), (192, 2), (1024, 1)],
    "0.5": [(48, 4), (96, 8), (192, 4), (1024, 1)],
    "1.0": [(116, 4), (232, 8), (464, 4), (1024, 1)],
    "1.5": [(176, 4), (352, 8), (704, 4), (1024, 1)],
    "2.0": [(244, 4), (488, 8), (976, 4), (2048, 1)],
}
stride_list = [2, 2, 2, 1, 1]


def build_model_backbone(in_x, model_scale="1.0", shuffle_group=2):
    num_of_feat = size2feat[model_scale]

    x = ConvLayer(
        kw=3, kh=3, in_f=in_x.get_shape()[-1], out_f=24, kernel_initializer=init_conv,
        use_bias=False, activation=None, stride=stride_list[0], name='conv1'
    )(in_x)
    x = BatchNormLayer(D=x.get_shape()[-1], name=f'bn_1')(x)
    x = ActivationLayer(activation=actv, name=f'activation_1')(x)
    x = MaxPoolLayer(name='maxpool1', ksize=[1, 3, 3, 1], strides=[1, stride_list[1], stride_list[1], 1])(x)

    for idx, (block, stride_single) in enumerate(zip(num_of_feat[:-1], stride_list[2:])):
        out_channel, repeat = block

        # First block is downsampling
        x = shuffle_net_spatial_down_samp_unit(
            x, out_channel, f"{idx}_block_down_shufflenet_",
            shuffle_group=shuffle_group, stride=stride_single
        )

        # Rest blocks
        for i in range(repeat - 1):
            x = shuffle_net_basic_unit(x, out_channel, f"{idx}_block_num_{i}_shufflenet_", shuffle_group=shuffle_group)

    x = ConvLayer(
        kw=1, kh=1, in_f=x.get_shape()[-1], out_f=num_of_feat[-1][0], kernel_initializer=init_conv,
        use_bias=False, activation=None, name='final'
    )(x)
    x = BatchNormLayer(D=x.get_shape()[-1], name=f'final_bn')(x)
    x = ActivationLayer(activation=actv, name=f'final_activation')(x)

    return in_x, x
