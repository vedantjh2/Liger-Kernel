import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous

@triton.jit
def _tvd_kernel(
    X_ptr, X_stride,
    Y_ptr, Y_stride,
    loss_ptr, loss_stride,
    dX_ptr, dX_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    X_ptr += pid * X_stride
    dX_ptr += pid * dX_stride
    Y_ptr += pid * Y_stride
    loss_ptr += pid * loss_stride

    row_loss = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        X = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        Y = tl.load(Y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        diff = tl.abs(Y - X)
        row_loss += tl.sum(diff, axis=0)
        
        dX = -0.5 * tl.where(X > Y, 1.0, -1.0) / n_cols
        tl.store(dX_ptr + offsets, dX, mask=mask)
    
    loss = 0.5 * row_loss / n_cols
    tl.store(loss_ptr, loss)

MAX_FUSED_SIZE = 65536

def tvd_forward(input, target):
    BT, V = input.shape
    n_rows = BT
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    loss = torch.empty(n_rows, dtype=torch.float32, device=input.device)
    dX = torch.empty_like(input)

    _tvd_kernel[(n_rows,)](
        X_ptr=input, X_stride=input.stride(0),
        Y_ptr=target, Y_stride=target.stride(0),
        loss_ptr=loss, loss_stride=loss.stride(0),
        dX_ptr=dX, dX_stride=dX.stride(0),
        n_rows=n_rows, n_cols=V,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    total_loss = loss.sum()
    return total_loss.to(input.dtype), dX.to(input.dtype)

def tvd_backward(dX, grad_output):
    return grad_output.view(-1, 1) * dX

class LigerTVDFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss, dX = tvd_forward(input, target)
        ctx.save_for_backward(dX)
        return loss

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (dX,) = ctx.saved_tensors
        return tvd_backward(dX, grad_output), None