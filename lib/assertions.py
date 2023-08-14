from typing import Tuple, Union

import tensorflow as tf


class InvalidTensorError(ValueError):
    def __init__(
        self,
        message="Error in rank and/or compatibility check. The input tensor should be a valid tf.Tensor.",
    ):
        super().__init__(message)


def assert_rank_and_dtype(
    tensor: tf.Tensor,
    rank: Union[int, Tuple[int]],
    dtype: Union[tf.dtypes.DType, Tuple[tf.dtypes.DType]],
):
    assert_rank(tensor, rank)
    assert_dtype(tensor, dtype)


def assert_rank(tensor: tf.Tensor, rank: Union[int, Tuple[int]]) -> None:
    if not isinstance(tensor, tf.Tensor):
        raise InvalidTensorError()
    supported_rank = []
    if isinstance(rank, tuple):
        supported_rank = rank
    else:
        supported_rank.append(rank)
    if len(tensor.shape) not in supported_rank:
        raise InvalidTensorError(
            f"Error in rank and/or compatibility check. The input tensor should be rank {rank} tf.Tensor, got {tensor.shape}."
        )


def assert_dtype(
    tensor: tf.Tensor, dtype: Union[tf.dtypes.DType, Tuple[tf.dtypes.DType]]
) -> None:
    if not isinstance(tensor, tf.Tensor):
        raise InvalidTensorError()
    supported_dtype = []
    if isinstance(dtype, tuple):
        supported_dtype = dtype
    else:
        supported_dtype.append(dtype)
    if tensor.dtype not in supported_dtype:
        raise InvalidTensorError(
            f"Error in rank and/or compatibility check. The input tensor should be {dtype}, got {tensor.dtype}."
        )


def assert_batch_dimension(tensor: tf.Tensor, batch_dize: int, dim: int = 0) -> None:
    if not isinstance(tensor, tf.Tensor):
        raise InvalidTensorError()

    if tensor.shape[dim] != batch_dize:
        raise InvalidTensorError(
            f"Error in rank and/or compatibility check. The input tensor should have {batch_dize} entry on batch dimension {dim}, got {tensor.shape}."
        )
