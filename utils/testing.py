import torch
def assert_almost_equal(x, y, decimal=2, err_msg=''):
    import numpy.testing as npt
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    npt.assert_array_almost_equal(x, y, err_msg=err_msg, decimal=decimal)