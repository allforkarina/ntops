"""
Arrangement for cyclic shift (roll) operations.

Design rationale:
  The three built-in arrangements (element_wise, reduction, pooling) all assume
  that each GPU block's computation is self-contained — all data needed by a
  block resides within that block.  Roll violates this assumption at the
  wrap-around boundary: the first block along the roll dimension needs elements
  from the *end* of the dimension, which belong to a different block.

  The arrangement below solves this by giving each block the **full** roll
  dimension (tile shape uses ``-1`` = "keep whole"), while tiling only the
  non-roll dimensions for parallelism::

      input  shape:  (B, S, H)          roll dim = H
      permute:        (B, S, H)          (H already last)
      flatten:        (B*S, H)
      tile:           (block_size, -1)  → each block gets (block_size, H)

  Each block then holds all H elements for a subset of the (B,S) positions.
  The roll is computed **within the application** by iterating the full
  dimension without any cross-block communication.

Limitation:
  The roll kernel uses this arrangement to verify whether ninetoothed supports
  computed index expressions such as ``input[j, shift + i]`` inside the
  application function.
"""

import ninetoothed


def arrangement(input, output, shift, dim, block_size=None):
    """
    Arrange input and output tensors for a roll (cyclic shift) along *dim*.

    Each block receives the entire *dim* dimension, while the remaining
    dimensions are flattened and tiled by *block_size* for GPU parallelism.

    Args:
        input:      Source tensor descriptor (ninetoothed.Tensor).
        output:     Destination tensor descriptor.
        dim:        The dimension along which the roll occurs.
        block_size: Number of non-roll positions per GPU block.
                    If None, inferred via ``ninetoothed.block_size()``.

    Returns:
        Tuple (input_arranged, output_arranged).
    """
    ndim = input.ndim

    if block_size is None:
        block_size = ninetoothed.block_size()

    # ── 1. Permute so the roll dimension is last ──────────────────────
    non_roll_dims = tuple(i for i in range(ndim) if i != dim)
    perm_order = non_roll_dims + (dim,)

    input_arranged = input.permute(perm_order)
    output_arranged = output.permute(perm_order)

    # ── 2. Flatten all non-roll dimensions into one ───────────────────
    if len(non_roll_dims) > 1:
        input_arranged = input_arranged.flatten(end_dim=len(non_roll_dims) - 1)
        output_arranged = output_arranged.flatten(end_dim=len(non_roll_dims) - 1)
    elif len(non_roll_dims) == 0:
        input_arranged = input_arranged[None, :]
        output_arranged = output_arranged[None, :]

    # Shape is now  (total_non_roll, dim_size)

    # ── 3. Tile non-roll dimension; keep roll dimension whole ─────────
    input_arranged = input_arranged.tile((block_size, -1))
    output_arranged = output_arranged.tile((block_size, -1))

    return input_arranged, output_arranged, shift
