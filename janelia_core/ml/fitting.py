""" Tools for fitting ML models.  """

import torch


def match_torch_module(tgt_m: torch.nn.Module, fit_m: torch.nn.Module, dim_ranges: torch.Tensor,
                       optim_opts: dict = None, n_its: int = 1000, n_smps:int = 100, device: torch.device = None,
                       update_int: int = 100) -> torch.Tensor:
    """ Optimizes one torch module to match another.

    Args:

        tgt_m: The target torch module whose input to output relationship we want to match

        fit_m: The module we seek to optimize

        dim_ranges: dim_ranges[:,i] gives the lower bound (1st entry) and upper bound (2nd entry) of values
        for input dimension i

        optim_opts_opts: A dictionary of options to use when creating the Adam optimizer.  These will be passed
        directly into the constructor.  If None, an empty dictionary will be created.

        n_its: The number of fitting iterations to perform

        n_smps: The number of random samples to generate each fitting iteration

        device: The device to perform optimization on. If none, optimization will be performed on cpu.

        update_int: The interval, in iterations, at which we print fitting status

    Returns:

        er: The objective at the end of optimization


    """

    if device is None:
        device = torch.device('cpu')

    if optim_opts is None:
        optim_opts = dict()

    # Get basic information
    dim_ranges = dim_ranges.to(device)
    n_dims = dim_ranges.shape[1]
    dim_span = dim_ranges[1, :] - dim_ranges[0, :]

    # Move target and fit modules to the specified device
    tgt_m.to(device)
    fit_m.to(device)

    params = fit_m.parameters()
    optimizer = torch.optim.Adam(params=params, **optim_opts)
    for i in range(n_its):

        # Generate samples
        with torch.no_grad():
            x = torch.rand([n_smps, n_dims], device=device)*dim_span + dim_ranges[0, :]
            y = tgt_m(x)

        # Run one step of optimization
        optimizer.zero_grad()
        y_hat = fit_m(x)
        er = torch.sum((y - y_hat)**2)

        if (i % update_int == 0) and (update_int < n_its):
            print('It: ' + str(i) + ', Error: ' + '{:e}'.format(er))

        er.backward()
        optimizer.step()

    return er
