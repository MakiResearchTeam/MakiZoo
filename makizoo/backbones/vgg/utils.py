
def get_batchnorm_params():
    return {
            'decay': 0.9,
            'eps': 1e-3
    }

def get_maxpool_params():
    return {
        'ksize': [1,2,2,1],
        'strides': [1,2,2,1],
        'padding': 'SAME'
    }


