import torch

def _circular_pad(inputs, padding):
    """
    Computation of circular padding.

    Arguments:
    ----------
    inputs: a (d+2)-order (d <= 3) tensor of size
        [batch_size, num_channels, input_size_1, ..., input_size_d] 
        The input before circular padding.

    padding: a tuple of ints with length 2*d (d <= 3)
        (padding_front_1, padding_rear_1, ..., padding_front_d, padding_rear_d)
        The number of entries added to each side of the inputs.

    Return:
    -------
    outputs: a (d+2)-order (d <= 3) tensor of size
        [batch_size, num_channels, output_size_1, ..., output_size_d]
        The output after circular padding.
        Note: output_size_i = input_size_i + padding_front_i + padding_rear_i

    """
    if padding[-1] > 0:
        inputs = torch.cat([inputs, inputs[:, :, 0:padding[-1]]], dim = 2)
        inputs = torch.cat([inputs[:, :, -(padding[-1] + padding[-2]):-padding[-1]], inputs], dim = 2)
    elif padding[-2] > 0:
        inputs = torch.cat([inputs[:, :, -padding[-2]:], inputs], dim = 2)

    if len(padding) > 2:
        if padding[-3] > 0:
            inputs = torch.cat([inputs, inputs[:, :, :, 0:padding[-3]]], dim = 3)
            inputs = torch.cat([inputs[:, :, :, -(padding[-3] + padding[-4]):-padding[-3]], inputs], dim = 3)
        elif padding[-4] > 0:
            inputs = torch.cat([inputs[:, :, :, -padding[-4]:], inputs], dim = 3)

    if len(padding) > 4:
        if padding[-5] > 0:
            inputs = torch.cat([inputs, inputs[:, :, :, :, 0:padding[-5]]], dim = 4)
            inputs = torch.cat([inputs[:, :, :, :, -(padding[-5] + padding[-6]):-padding[-5]], inputs], dim = 4)
        elif padding[-5] > 0:
            inputs = torch.cat([inputs[:, :, :, :, -padding[-6]:], inputs], dim = 4)

    return inputs
