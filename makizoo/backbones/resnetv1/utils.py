
def get_batchnorm_params():
    return {
            'decay': 0.9,
            'eps': 1e-3
    }

def get_head_batchnorm_params():
    return {
            'use_gamma': False,
            'decay': 0.9,
            'eps': 1e-3
    }
