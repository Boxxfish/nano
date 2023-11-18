"""
This experiment sees whether NCA can learn to use its environment to produce
complex effects.
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
before_path = "temp/cover/before.png"
img_path = "temp/cover/after.png"
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


def perform_update(curr_state: torch.Tensor, net: nn.Module, static_mask: torch.Tensor) -> torch.Tensor:
    """
    Performs one update step, returning the next state.
    """
    max_a = torch.max_pool2d(curr_state[:, 3, :, :], 3, padding=1, stride=1).unsqueeze(
        1
    )  # Size: (1, 1, sim_size, sim_size)
    mask_a = max_a > min_a_alive
    update = net(curr_state)
    update_mask = (
        torch.distributions.Uniform(0.0, 1.0)
        .sample(torch.Size((sim_size, sim_size)))
        .unsqueeze(0)
        .unsqueeze(0)
    ).to(device) < cell_update # Shape: (1, 1, sim_size, sim_size)
    curr_state = curr_state + update * update_mask * static_mask
    curr_state = curr_state * mask_a
    return curr_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    # Load image
    seed_arrs = []
    final_imgs = []
    for _ in range(10):
        img = Image.open(img_path)
        img_before = Image.open(before_path)
        size = random.randrange(4, 10) / 10
        new_w = int(sim_size // 2 * size)
        new_h = int(sim_size // 2 * size)
        paste_x = 0 if random.random() < 0.5 else sim_size - new_w
        paste_y = 0 if random.random() < 0.5 else sim_size - new_h
        if random.random() < 0.5:
            paste_x = int((sim_size - new_w) * random.random())
        else:
            paste_y = int((sim_size - new_h) * random.random())
        img = img.resize((new_w, new_h)).rotate(random.randrange(0, 360))
        final_img = Image.new("RGBA", (sim_size, sim_size))
        final_img.paste(img, (paste_x, paste_y))
        final_img_arr = np.array(final_img).transpose((2, 0, 1)) / 255
        final_imgs.append(final_img_arr)

        # Set up initial seed image
        img_before = img_before.resize((new_w, new_h)).rotate(random.randrange(0, 360))
        seed_img = Image.new("RGBA", (sim_size, sim_size))
        seed_img.paste(img_before, (paste_x, paste_y))
        seed_img_arr = np.array(seed_img).transpose((2, 0, 1)) / 255
        seed_arr = np.concatenate([seed_img_arr, np.zeros([state_size - 4, sim_size, sim_size])], 0)
        seed_arr[:4] = seed_img_arr
        seed_arrs.append(seed_arr)
    seed_arrs = np.stack(seed_arrs, 0)
    final_imgs = np.stack(final_imgs, 0)

    static_masks = torch.from_numpy(seed_arrs[:, 3, :, :] <= 0.5).float().to(device) # 0 if the cell should be not be updated 

    if args.eval:
        net = UpdateNet(state_size).to(device)
        net.load_state_dict(torch.load("temp/net.pt"))

        with torch.no_grad():
            idx = random.randrange(0, seed_arrs.shape[0])
            curr_state = torch.from_numpy(seed_arrs[idx]).unsqueeze(0).to(device)
            for i in tqdm(range(max_train)):
                curr_state = perform_update(curr_state, net, static_masks[idx])
                ax = plt.subplot()
                ax.set_facecolor("gray")
                ax.imshow(np.clip(curr_state[0][:4].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0))
                plt.savefig(f"temp/generated/cover/{i}.png")
        quit()

    # Train
    net = UpdateNet(state_size).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    for i in tqdm(range(train_iters)):
        idx = random.randrange(0, seed_arrs.shape[0])
        final = torch.from_numpy(final_imgs[idx]).unsqueeze(0).to(device)
        curr_state = (
            torch.from_numpy(seed_arrs[idx]).unsqueeze(0).to(device)
        )  # Shape: (1, state_size, sim_size, sim_size)
        for _ in range(random.randrange(min_train, max_train)):
            curr_state = perform_update(curr_state, net, static_masks[idx])

        cmp_state = curr_state[:, :4, :, :]
        opt.zero_grad()
        loss = ((cmp_state - final) ** 2).mean()
        loss.backward()
        tqdm.write(f"Loss: {loss.item()}")
        opt.step()

        # Save network
        if (i + 1) % 100 == 0:
            torch.save(net.cpu().state_dict(), "temp/net.pt")
            net.to(device)


if __name__ == "__main__":
    main()
