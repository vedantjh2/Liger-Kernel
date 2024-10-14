import torch
import triton
import triton.language as tl


@triton.jit
def tvd_kernel(
    p_ptr,
    q_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(p_ptr + offsets, mask=mask)
    q = tl.load(q_ptr + offsets, mask=mask)

    diff = tl.abs(p - q)
    sum_diff = tl.sum(diff, axis=0)

    if pid == 0:
        tl.store(output_ptr, sum_diff * 0.5)


class LigerTVDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, q):
        assert p.is_cuda and q.is_cuda, "Input tensors must be on GPU"
        assert p.shape == q.shape, "Input tensors must have the same shape"
        assert p.dtype == q.dtype, "Input tensors must have the same dtype"

        ctx.save_for_backward(p, q)

        output = torch.empty((), dtype=p.dtype, device=p.device)
        n_elements = p.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        tvd_kernel[grid](p.flatten(), q.flatten(), output, n_elements, BLOCK_SIZE=1024)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        p, q = ctx.saved_tensors
        grad_p = grad_q = None

        if ctx.needs_input_grad[0]:
            grad_p = 0.5 * grad_output * torch.sign(p - q)
        if ctx.needs_input_grad[1]:
            grad_q = -0.5 * grad_output * torch.sign(p - q)

        return grad_p, grad_q


def liger_tvd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return LigerTVDFunction.apply(p, q)
