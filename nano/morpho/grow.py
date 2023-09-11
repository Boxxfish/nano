"""
This experiment replicates experiment 1 ("Grow") from the Growing Neural
Cellular Automata post.
"""
import argparse
import random
from PIL import Image  # type: ignore
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm

# Params
img_path = "assets/platformer/p1.png"
min_train = 64  # Minimum number of training steps.
max_train = 96  # Maximum number of training steps.
sim_size = 64  # Size of each dim of sim.
state_size = 16  # Number of states in a cell, including RGBA.
cell_update = 0.5  # Probability of a cell being updated.
train_iters = 4000  # Number of total training iterations.
min_a_alive = 0.1  # To be considered alive, a cell in the neighborhood's alpha value must be at least this.
device = torch.device("cuda")

class UpdateNet(nn.Module):
    """
    Given the current sim state, returns the update rule for each cell.
    """

    def __init__(self, state_size: int):
        super().__init__()
        out_conv = nn.Conv2d(128, state_size, 1, bias=False)
        out_conv.weight = nn.Parameter(torch.zeros(out_conv.weight.shape))
        self.net = nn.Sequential(
            nn.Conv2d(state_size * 3, 128, 1, bias=False),
            nn.ReLU(),
            out_conv,
        )
        self.state_size = state_size
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        self.sobel_x = nn.Parameter(
            sobel_x.unsqueeze(0).unsqueeze(0).repeat(state_size, 1, 1, 1),
            requires_grad=False,
        )
        self.sobel_y = nn.Parameter(
            sobel_x.T.unsqueeze(0).unsqueeze(0).repeat(state_size, 1, 1, 1),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute perception vectors
        grad_x = nn.functional.conv2d(
            x.float(), self.sobel_x, groups=self.state_size, padding=1
        )
        grad_y = nn.functional.conv2d(
            x.float(), self.sobel_y, groups=self.state_size, padding=1
        )
        x = torch.concatenate([x.float(), grad_x, grad_y], 1)

        # Compute update
        update = self.net(x)
        return update


def perform_update(curr_state: torch.Tensor, net: nn.Module) -> torch.Tensor:
    """
    Performs one update step, returning the next state.
    """
    update = net(curr_state)
    update_mask = (
        torch.distributions.Uniform(0.0, 1.0)
        .sample(torch.Size((sim_size, sim_size)))
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(device)  # Shape: (1, 1, sim_size, sim_size)
    curr_state = curr_state + update * update_mask
    max_a = torch.max_pool2d(curr_state[:, 3, :, :], 3, padding=1, stride=1).unsqueeze(
        1
    )  # Size: (1, 1, sim_size, sim_size)
    mask_a = max_a > min_a_alive
    curr_state = curr_state * mask_a
    return curr_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    # Load image
    img = Image.open(img_path)
    img_w, img_h = img.size
    max_size = max(img_w, img_h)
    new_w = int(sim_size / 2 * (img_w / max_size))
    new_h = int(sim_size / 2 * (img_h / max_size))
    paste_x = sim_size // 2 - new_w // 2
    paste_y = sim_size // 2 - new_h // 2
    img = img.resize((new_w, new_h))
    final_img = Image.new("RGBA", (sim_size, sim_size))
    final_img.paste(img, (paste_x, paste_y))
    final_img_arr = np.array(final_img).transpose((2, 0, 1)) / 255

    # Set up initial seed image
    seed_arr = np.zeros((state_size, sim_size, sim_size))
    for i in range(3, state_size):
        seed_arr[i][sim_size // 2][sim_size // 2] = 1.0

    if args.eval:
        net = UpdateNet(state_size).to(device)
        net.load_state_dict(torch.load("temp/net.pt"))

        with torch.no_grad():
            curr_state = torch.from_numpy(seed_arr).unsqueeze(0).to(device)
            for i in tqdm(range(64)):
                curr_state = perform_update(curr_state, net)
                plt.imshow(np.clip(curr_state[0][:4].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0))
                plt.savefig(f"temp/generated/grow/{i}.png")
        quit()

    # Train
    net = UpdateNet(state_size).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    final = torch.from_numpy(final_img_arr).unsqueeze(0).to(device)
    for i in tqdm(range(train_iters)):
        curr_state = (
            torch.from_numpy(seed_arr).unsqueeze(0).to(device)
        )  # Shape: (1, state_size, sim_size, sim_size)
        for _ in range(random.randrange(min_train, max_train)):
            curr_state = perform_update(curr_state, net)

        cmp_state = curr_state[:, :4, :, :]
        opt.zero_grad()
        loss = ((cmp_state - final) ** 2).mean()
        loss.backward()
        tqdm.write(f"Loss: {loss.item()}")
        opt.step()

    # Save network
    torch.save(net.cpu().state_dict(), "temp/net.pt")


if __name__ == "__main__":
    main()
