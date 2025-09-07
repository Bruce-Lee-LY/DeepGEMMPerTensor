import torch
from typing import List, Tuple

from ..jit import build
from .runtime import (
    FP8WGradGemmRuntime, GemmType,
    make_2d_tma_a_desc, make_2d_tma_b_desc,
    make_2d_tma_d_desc)
from .gemm_per_tensor import get_best_configs
from .utils import ceil_div, get_num_sms


def wgrad_gemm_per_tensor_fp8_fp8_fp32_nt(lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor):
    """
    Perform a weight gradient GEMM with FP8 inputs and FP32 output.
        Results will be accumulated into the output tensor.

    Requirements:
        LHS, RHS, and output tensors must be contiguous in dimension 1, i.e., stride(1) = 1.
        The stride(0) of LHS and RHS must be a multiple of 16, and the stride(0) of output must be a multiple of 4.

    Arguments:
        lhs: an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`.
        rhs: an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`.
        out: the FP32 output tensor of shape `[m, n]`, which will be accumulated.
    """
    m, k = lhs.shape
    n, k_ = rhs.shape
    m_, n_ = out.shape

    # Type and shape checks
    assert m == m_ and n == n_ and k == k_
    assert n > 0 and m > 0
    assert lhs.dtype == torch.float8_e4m3fn
    assert rhs.dtype == torch.float8_e4m3fn
    assert out.dtype == torch.float
    assert lhs.stride(1) == 1 and out.stride(1) == 1 and rhs.stride(1) == 1

    # Do nothing if `k` is zero
    if k == 0:
        return

    # K must be aligned to 128
    aligned_k = ceil_div(k, 128) * 128

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = get_best_configs(
        m, n, aligned_k, 1, num_sms, is_fp32_out=True, is_wgrad=True)
    num_last_stages = ceil_div(k, 128) % num_stages
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    tensor_map_a = make_2d_tma_a_desc(
        GemmType.Normal, lhs, m, k, lhs.stride(0), block_m, block_k, 1)
    tensor_map_b = make_2d_tma_b_desc(
        GemmType.Normal, rhs, n, k, rhs.stride(0), block_n, block_k, 1)
    tensor_map_d = make_2d_tma_d_desc(
        GemmType.Normal, out, m, n, out.stride(0), block_m, block_n, 1, smem_config[1])

    kwargs = {
        # Templated arguments
        'GEMM_TYPE': GemmType.Normal,
        'NUM_TMA_THREADS': num_tma_threads,
        'NUM_MATH_THREADS_PER_GROUP': num_math_threads_per_group,
        'M': m, 'N': n, 'K': aligned_k,
        'NUM_GROUPS': 1,
        'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k,
        'NUM_STAGES': num_stages,
        'NUM_LAST_STAGES': num_last_stages,
        'NUM_TMA_MULTICAST': tma_multicast_config[0],
        'IS_TMA_MULTICAST_ON_A': tma_multicast_config[1],
        # Runtime arguments
        'NUM_SMS': num_sms,
        'SMEM_SIZE': smem_config[0],
        'TENSOR_MAP_A': tensor_map_a,
        'TENSOR_MAP_B': tensor_map_b,
        'TENSOR_MAP_D': tensor_map_d,
        'STREAM': torch.cuda.current_stream().cuda_stream,
        'DEVICE_INDEX': out.device.index
    }

    # Generate, build and run the kernel
    code = FP8WGradGemmRuntime.generate(kwargs)
    runtime = build('wgrad_gemm_per_tensor_fp8_fp8_fp32_nt',
                    code, FP8WGradGemmRuntime, kwargs)
    runtime(**kwargs)


def k_grouped_wgrad_gemm_per_tensor_fp8_fp8_fp32_nt(lhs: torch.Tensor, rhs: torch.Tensor, out: torch.Tensor, batch_sizes: List[int]):
    """
    Perform a k-grouped weight gradient GEMM with FP8 inputs and FP32 output.
        Results will be accumulated into the output tensor.

    Requirements:
        This function handles multiple batches with varying k-dimensions, processing each batch sequentially.
        Each batch's LHS, RHS, and output tensors must be contiguous.

    Arguments:
        lhs: A flattened FP8 tensor (typed `torch.float8_e4m3fn`) containing all batches of LHS data,
                 and the flattened shape is `[sum(m * k for k in batch_sizes)]`, where m is the number of rows.
        rhs: A flattened FP8 tensor (typed `torch.float8_e4m3fn`) containing all batches of RHS data,
                 and the flattened shape is `[sum(n * k for k in batch_sizes)]`, where n is the number of rows.
        out: The FP32 output tensor of shape [num_batches, m, n], which will be accumulated.
        batch_sizes: A list of integers specifying the k-dimension for each batch.
    """
    num_batches, m, n = out.shape

    lhs_offset, rhs_offset = 0, 0

    for i in range(num_batches):
        k = batch_sizes[i]
        lhs_slice = lhs[lhs_offset:lhs_offset + m * k].view(m, k)
        rhs_slice = rhs[rhs_offset:rhs_offset + n * k].view(n, k)
        wgrad_gemm_per_tensor_fp8_fp8_fp32_nt(lhs_slice, rhs_slice, out[i])

        lhs_offset += m * k
        rhs_offset += n * k
