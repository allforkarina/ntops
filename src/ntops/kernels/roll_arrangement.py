"""Arrangement for cyclic shift (roll) operations."""


def arrangement(input, output, shift, roll_dim, block_size=None):
    if block_size is None:
        block_size = 1

    non_roll_dims = tuple(i for i in range(input.ndim) if i != roll_dim)
    perm_order = non_roll_dims + (roll_dim,)

    input_arranged = input.permute(perm_order)
    output_arranged = output.permute(perm_order)

    block_shape = tuple(1 for _ in non_roll_dims) + (-1,)

    input_arranged = input_arranged.tile(block_shape)
    output_arranged = output_arranged.tile(block_shape)

    if non_roll_dims:
        non_roll_indices = tuple(range(len(non_roll_dims)))
        input_arranged.dtype = input_arranged.dtype.squeeze(non_roll_indices)
        output_arranged.dtype = output_arranged.dtype.squeeze(non_roll_indices)

    return input_arranged, output_arranged, shift
