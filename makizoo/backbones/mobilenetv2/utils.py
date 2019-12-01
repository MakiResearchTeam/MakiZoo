

# This function taken from original git of MobileNetV2
# Main idea of these function - make numbers (`v` in function)
# divisible by other number (`divisor` in function) without remainder
def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def get_batchnorm_params():
    return {
            'decay': 0.9,
            'eps': 1e-3
    }
