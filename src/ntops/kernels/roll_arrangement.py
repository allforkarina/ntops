"""Arrangement for cyclic shift (roll) operations."""


def arrangement(input, output, shift, roll_dim, block_size=None):
    if block_size is None:
        block_size = 1

    non_roll_dims = tuple(i for i in range(input.ndim) if i != roll_dim)
    perm_order = non_roll_dims + (roll_dim,)

    input_arranged = input.permute(perm_order)
    output_arranged = output.permute(perm_order)

    if len(non_roll_dims) > 1:
        input_arranged = input_arranged.flatten(end_dim=len(non_roll_dims) - 1)
        output_arranged = output_arranged.flatten(end_dim=len(non_roll_dims) - 1)
    elif len(non_roll_dims) == 0:
        input_arranged = input_arranged[None, :]
        output_arranged = output_arranged[None, :]

    return (
        input_arranged.tile((block_size, -1)),
        output_arranged.tile((block_size, -1)),
        shift,
    )
