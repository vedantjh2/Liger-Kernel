import pytest
import torch
from torch.nn import functional as F

from liger_kernel.transformers.functional import liger_tvd
from liger_kernel.transformers.tvd import LigerTVD, LigerTVDFunction

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class TVD(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float):
        super(TVD, self).__init__()
        self.dtype = dtype

    def forward(self, q: torch.Tensor, p: torch.Tensor):
        p, q = p.to(torch.float32), q.to(torch.float32)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        loss = 0.5 * torch.sum(torch.abs(p - q), dim=-1).mean()
        return loss.to(self.dtype)

_SHAPE_PARAMS = (
    "B, T, V",
    [
        (2, 512, 4000),  # Reduced size
        (2, 512, 4000),  # Reduced size
        (41, 101, 1271),  # Original size
        (1, 512, 16000),  # Reduced size
        (3, 123, 4000),   # Reduced size
    ],
)

_DTYPE_PARAMS = (
    "dtype, atol, rtol",
    [
        (torch.bfloat16, 1e-2, 1e-1),
        (torch.float32, 1e-4, 1e-3),
        (torch.float16, 1e-2, 1e-1),
    ],
)

def assert_verbose_allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08, max_print=5):
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff)
        max_diff_index = torch.argmax(diff)
        print(f"Max difference: {max_diff.item()} at index {max_diff_index.item()}")
        print(f"tensor1 value: {tensor1.flatten()[max_diff_index].item()}")
        print(f"tensor2 value: {tensor2.flatten()[max_diff_index].item()}")
        assert False, "Tensors are not close enough"

def _test_correctness_once(
    target_tvd, B, T, V, dtype, atol, rtol, is_last_layer=True, device="cuda"
):
    torch_tvd = TVD(dtype=dtype)

    input = torch.randn(
        B * T, V, device=device, dtype=dtype, requires_grad=True
    ).softmax(dim=-1)
    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).softmax(dim=-1)

    output = torch_tvd(x1, target)
    output2 = target_tvd(x2, target)

    assert_verbose_allclose(output, output2, atol=atol, rtol=rtol)

    if not is_last_layer:
        output = output * 2.0
        output2 = output2 * 2.0

    output.backward()
    output2.backward()

    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)

@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness(B, T, V, dtype, atol, rtol):
    liger_tvd = LigerTVD()
    _test_correctness_once(liger_tvd, B, T, V, dtype, atol, rtol)

@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
def test_correctness_not_last(B, T, V, dtype, atol, rtol):
    liger_tvd = LigerTVD()
    _test_correctness_once(liger_tvd, B, T, V, dtype, atol, rtol, is_last_layer=False)

@pytest.mark.parametrize(*_SHAPE_PARAMS)
@pytest.mark.parametrize(*_DTYPE_PARAMS)
@pytest.mark.parametrize("is_last_layer", [False, True])
def test_correctness_functional(B, T, V, is_last_layer, dtype, atol, rtol):
    _test_correctness_functional(B, T, V, is_last_layer, dtype, atol, rtol)

def _test_correctness_functional(B, T, V, is_last_layer, dtype, atol, rtol, device="cuda"):
    input = torch.randn(B * T, V, device=device, dtype=dtype, requires_grad=True).softmax(dim=-1)
    x1 = input.detach().clone().requires_grad_(True)
    x2 = input.detach().clone().requires_grad_(True)

    with torch.no_grad():
        target = torch.randn(B * T, V, dtype=dtype, device=device).softmax(dim=-1)

    output = LigerTVDFunction.apply(x1, target)
    output2 = liger_tvd(x2, target)
    
    assert_verbose_allclose(output, output2, atol=atol, rtol=rtol)
    
    if not is_last_layer:
        output = output * 2.0
        output2 = output2 * 2.0
    
    output.backward()
    output2.backward()
    
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)