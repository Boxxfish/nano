"""
This experiment replicates the StampCA blog post, using an autoencoder to encode
the state of the starting seed.
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
img_paths = [
    "assets/platformer/p1.png",
    "assets/platformer/p2.png",
    "assets/platformer/p3.png",
    "assets/platformer/p4.png",
]
min_train = 64  # Minimum number of training steps.
max_train = 96  # Maximum number of training steps.
sim_size = 64  # Size of each dim of sim.
state_size = 16  # Number of states in a cell, including RGBA.
cell_update = 0.5  # Probability of a cell being updated.
train_iters = 100000  # Number of total training iterations.
min_a_alive = 0.1  # To be considered alive, a cell in the neighborhood's alpha value must be at least this.
batch_size = 4
device = torch.device("cuda")

class EncoderNet(nn.Module):
    """
    Given an image, returns a seed.
    """
    def __init__(self, state_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, state_size - 4, 3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x.float())
        return torch.max(torch.max(x, 3).values, 2).values

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
    ).to(
        device
    ) < cell_update # Shape: (1, 1, sim_size, sim_size)
    curr_state = curr_state + update * update_mask
    curr_state = curr_state * mask_a
    return curr_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--pic", default=0)
    args = parser.parse_args()

    # Load images
    imgs = [Image.open(img_path) for img_path in img_paths]
    final_img_arrs = []
    for img in imgs:
        img_w, img_h = img.size
        max_size = max(img_w, img_h)
        new_w = int(sim_size / 2 * (img_w / max_size))
        new_h = int(sim_size / 2 * (img_h / max_size))
        paste_x = sim_size // 2 - new_w // 2
        paste_y = sim_size // 2 - new_h // 2
        img = img.resize((new_w, new_h))
        final_img = Image.new("RGBA", (sim_size, sim_size))
        final_img.paste(img, (paste_x, paste_y))
        final_img_arrs.append(np.array(final_img).transpose((2, 0, 1)) / 255)
    final_img_arrs = np.stack(final_img_arrs)

    if args.eval:
        net = UpdateNet(state_size).to(device)
        enc_net = EncoderNet(state_size).to(device)
        net.load_state_dict(torch.load("temp/net.pt"))
        enc_net.load_state_dict(torch.load("temp/enc_net.pt"))

        with torch.no_grad():
            final = torch.from_numpy(final_img_arrs[int(args.pic)]).unsqueeze(0).to(device)
            color = torch.zeros((1, 4), device=device)
            color[:, 3] = 1.0
            seed = torch.concatenate([color, enc_net(final)], 1)
            curr_state = torch.zeros((1, sim_size, sim_size, state_size), device=device)
            curr_state[0, sim_size // 2, sim_size // 2] = seed
            curr_state = curr_state.permute(0, 3, 1, 2)
            for i in tqdm(range(64)):
                curr_state = perform_update(curr_state, net)
                plt.imshow(
                    np.clip(curr_state[0][:4].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
                )
                plt.savefig(f"temp/generated/stamp/{i}.png")
        quit()

    # Train
    net = UpdateNet(state_size).to(device)
    enc_net = EncoderNet(state_size).to(device)
    opt = torch.optim.Adam(list(enc_net.parameters()) + list(net.parameters()), lr=0.001)
    for i in tqdm(range(train_iters)):
        img_idxs = list(range(final_img_arrs.shape[0]))
        random.shuffle(img_idxs)
        img_idxs = img_idxs[:batch_size]
        final = torch.from_numpy(final_img_arrs[np.array(img_idxs)]).to(device)
        color = torch.zeros((batch_size, 4), device=device)
        color[:, 3] = 1.0
        seed = torch.concatenate([color, enc_net(final)], 1)
        curr_state = torch.zeros((batch_size, sim_size, sim_size, state_size), device=device)
        curr_state[:, sim_size // 2, sim_size // 2] = seed
        curr_state = curr_state.permute(0, 3, 1, 2)
        for _ in range(random.randrange(min_train, max_train)):
            curr_state = perform_update(curr_state, net)

        cmp_state = curr_state[:, :4, :, :]
        opt.zero_grad()
        loss = ((cmp_state - final) ** 2).mean()
        loss.backward()
        tqdm.write(f"Loss: {loss.item()}")
        opt.step()

        if (i + 1) % 100 == 0:
            # Save networks
            torch.save(net.cpu().state_dict(), "temp/net.pt")
            torch.save(enc_net.cpu().state_dict(), "temp/enc_net.pt")
            net.to(device)
            enc_net.to(device)

    # Save networks
    torch.save(net.cpu().state_dict(), "temp/net.pt")
    torch.save(enc_net.cpu().state_dict(), "temp/enc_net.pt")


if __name__ == "__main__":
    main()
