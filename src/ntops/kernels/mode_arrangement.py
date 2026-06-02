def arrangement(input, values, indices, dim, block_size=None):
    if block_size is None:
        block_size = -1

    reduce_dim = dim if dim >= 0 else dim + input.ndim
    non_reduce_dims = tuple(i for i in range(input.ndim) if i != reduce_dim)

    input_arranged = input.permute(non_reduce_dims + (reduce_dim,))
    input_arranged = input_arranged.tile(tuple(1 for _ in non_reduce_dims) + (block_size,))

    if non_reduce_dims:
        non_reduce_indices = tuple(range(len(non_reduce_dims)))
        input_arranged.dtype = input_arranged.dtype.squeeze(non_reduce_indices)

    if values.ndim == 0:
        return input_arranged, values, indices

    block_shape = tuple(1 for _ in range(values.ndim))
    value_arranged = values.tile(block_shape)
    index_arranged = indices.tile(block_shape)
    output_indices = tuple(range(values.ndim))
    value_arranged.dtype = value_arranged.dtype.squeeze(output_indices)
    index_arranged.dtype = index_arranged.dtype.squeeze(output_indices)

    return input_arranged, value_arranged, index_arranged
