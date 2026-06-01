"""Arrangement for cyclic shift (roll) operations.

Roll is a data movement operator: each output element reads from a different
input coordinate.  This arrangement keeps the original tensor dimensions intact
so the application can use ``output.offsets(dim)`` to compute the corresponding
source coordinate.  Only the last dimension is tiled for parallelism, which
keeps the innermost program size bounded by ``block_size``.
"""

import ninetoothed


def arrangement(input, output, shift, dim, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()

    tile_shape = tuple(
        block_size if axis == input.ndim - 1 else 1 for axis in range(input.ndim)
    )

    return input.tile(tile_shape), output.tile(tile_shape), shift, dim
