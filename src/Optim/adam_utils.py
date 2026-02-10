"""Optim/adam_utils.py: Provides various utility functions for the Adam optimizer and its variants."""

import torch


def replace_param_group_data(optimizer: torch.optim.Optimizer, new_values: torch.Tensor, group_name: str, reset_state: bool = True) -> None:
    """Replaces the data of a parameter group with the given tensor."""
    for group in optimizer.param_groups:
        if group['name'] == group_name:
            if len(group['params']) != 1:
                raise NotImplementedError('"replace_param_group_data" only implemented for single-parameter groups.')
            param = group['params'][0]
            param.data = new_values
            if reset_state:
                state = optimizer.state[param]
                if state:
                    for val in ['exp_avg', 'exp_avg_sq']:
                        state[val].zero_()


def prune_param_groups(optimizer: torch.optim.Optimizer, mask: torch.Tensor, group_names: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Removes parameter entries based on the given mask."""
    new_params = {}
    for group in optimizer.param_groups:
        if group_names is not None and group['name'] not in group_names:
            continue
        if len(group['params']) != 1:
            raise NotImplementedError('"prune_param_groups" only implemented for single-parameter groups.')
        old_param = group['params'][0]
        state = optimizer.state[old_param]
        new_param = torch.nn.Parameter(old_param[mask])
        if state:
            for val in ['exp_avg', 'exp_avg_sq']:
                state[val] = state[val][mask]
            optimizer.state.pop(old_param)
            optimizer.state[new_param] = state
        group['params'][0] = new_param
        new_params[group['name']] = new_param
    return new_params


def extend_param_groups(optimizer: torch.optim.Optimizer, additional_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extend existing parameters by concatenating the given tensors."""
    new_params = {}
    for group in optimizer.param_groups:
        if len(group['params']) != 1:
            raise NotImplementedError('"extend_param_groups" only implemented for single-parameter groups.')
        extension_tensor = additional_params.get(group['name'], None)
        if extension_tensor is None:
            continue
        old_param = group['params'][0]
        state = optimizer.state[old_param]
        new_param = torch.nn.Parameter(torch.cat((old_param, extension_tensor), dim=0))
        if state:
            for val in ['exp_avg', 'exp_avg_sq']:
                state[val] = torch.cat((state[val], torch.zeros_like(extension_tensor)), dim=0)
            optimizer.state.pop(old_param)
            optimizer.state[new_param] = state
        group['params'][0] = new_param
        new_params[group['name']] = new_param
    return new_params


def reset_state(optimizer: torch.optim.Optimizer, group_names: list[str] | None = None, indices: torch.Tensor | None = None) -> None:
    """Resets the optimizer state for the specified parameter groups. If indices are provided, only those entries are reset."""
    for group in optimizer.param_groups:
        if group_names is not None and group['name'] not in group_names:
            continue
        if len(group['params']) != 1:
            raise NotImplementedError('"reset_state" only implemented for single-parameter groups.')
        param = group['params'][0]
        state = optimizer.state[param]
        if state:
            for val in ['exp_avg', 'exp_avg_sq']:
                if indices is not None:
                    state[val][indices] = 0
                else:
                    state[val].zero_()

def sort_param_groups(optimizer: torch.optim.Optimizer, ordering: torch.Tensor, group_names: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Sorts parameter entries based on the given ordering."""
    new_params = {}
    for group in optimizer.param_groups:
        if group_names is not None and group['name'] not in group_names:
            continue
        if len(group['params']) != 1:
            raise NotImplementedError('"sort_param_groups" only implemented for single-parameter groups.')
        old_param = group['params'][0]
        state = optimizer.state[old_param]
        new_param = torch.nn.Parameter(old_param[ordering])
        if state:
            for val in ['exp_avg', 'exp_avg_sq']:
                state[val] = state[val][ordering]
            optimizer.state.pop(old_param)
            optimizer.state[new_param] = state
        group['params'][0] = new_param
        new_params[group['name']] = new_param
    return new_params
