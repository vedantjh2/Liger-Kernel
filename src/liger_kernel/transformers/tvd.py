import torch.nn as nn

from liger_kernel.ops.tvd import LigerTVDFunction


class LigerTVD(nn.Module):
    r"""The Total Variation Distance.
    .. math::
        TVD(P, Q) = (1/2) * sum(|P - Q|)

    .. note::
        This function expects both arguments to be in probability space, not log-space.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: a scalar.

    Examples:
    ```python
    >>> tvd = LigerTVD()
    >>> # input should be a distribution in the probability space
    >>> input = torch.randn(3, 5, requires_grad=True).softmax(dim=-1)
    >>> target = torch.randn(3, 5, requires_grad=True).softmax(dim=-1)
    >>> output = tvd(input, target)
    ```
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return LigerTVDFunction.apply(input, target)
