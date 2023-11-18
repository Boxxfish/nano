"""
Shared functions across experiments.
"""

import torch
from torch import nn

def perform_update(
    curr_state: torch.Tensor,
    dna: torch.Tensor,
    net: nn.Module,
    min_a_alive: float,
    sim_size: int,
    cell_update: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Performs one update step, returning the next state.
    """
    max_a = torch.max_pool2d(curr_state[:, 3, :, :], 3, padding=1, stride=1).unsqueeze(
        1
    )  # Size: (1, 1, sim_size, sim_size)
    mask_a = (max_a <= min_a_alive).expand(curr_state.shape)
    update = net(curr_state, dna)
    update_mask = (
        torch.distributions.Uniform(0.0, 1.0)
        .sample(torch.Size((sim_size, sim_size)))
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(
        device
    ) > cell_update  # Shape: (1, 1, sim_size, sim_size)
    update_mask = update_mask.expand(curr_state.shape)
    update[update_mask] = 0
    curr_state = curr_state + update
    curr_state[mask_a] = 0
    return curr_state