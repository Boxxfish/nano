"""
This experiment extends the Neural DNA experiment to incorporate persistence via
a pool of previous states. Note that this experiment doesn't keep track of the
final image of each item in the pool, in theory allowing any configuration of
cells to transform into any other configuration.
"""
import argparse
import random
from PIL import Image  # type: ignore
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm
from pathlib import Path

from nano.morpho.shared import perform_update

# Params
base_path = Path("temp/lizard_skeleton")
img_paths = [path for path in base_path.iterdir()]
min_train = 64  # Minimum number of training steps.
max_train = 96  # Maximum number of training steps.
sim_size = 64  # Size of each dim of sim.
state_size = 16  # Number of states in a cell, including RGBA.
dna_size = 16  # Number of elements in the DNA vector.
cell_update = 0.5  # Probability of a cell being updated.
train_iters = 100000  # Number of total training iterations.
min_a_alive = 0.1  # To be considered alive, a cell in the neighborhood's alpha value must be at least this.
pool_size = 1024  # Number of items in the pool.
batch_size = 8
device = torch.device("cuda")


class EncoderNet(nn.Module):
    """
    Given an image, returns a DNA sequence.
    """

    def __init__(self, dna_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, dna_size, 3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x.float())
        return torch.max(torch.max(x, 3).values, 2).values


class UpdateNet(nn.Module):
    """
    Given the current sim state, returns the update rule for each cell.
    """

    def __init__(self, state_size: int, dna_size: int):
        super().__init__()
        out_conv = nn.Conv2d(128, state_size, 1, bias=False)
        out_conv.weight = nn.Parameter(torch.zeros(out_conv.weight.shape))
        self.net = nn.Sequential(
            nn.Conv2d(state_size * 3 + dna_size, 128, 1, bias=False),
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

    def forward(self, x: torch.Tensor, dna: torch.Tensor) -> torch.Tensor:
        # Compute perception vectors
        grad_x = nn.functional.conv2d(
            x.float(), self.sobel_x, groups=self.state_size, padding=1
        )
        grad_y = nn.functional.conv2d(
            x.float(), self.sobel_y, groups=self.state_size, padding=1
        )
        x = torch.concatenate([x.float(), grad_x, grad_y, dna.float()], 1)

        # Compute update
        update = self.net(x)
        return update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--pic", default=0)
    parser.add_argument("--t1", default=None)
    parser.add_argument("--t2", default=None)
    args = parser.parse_args()

    # Load images
    imgs = [Image.open(img_path) for img_path in img_paths]
    final_img_arrs = []
    for img in imgs:
        img_w, img_h = img.size
        max_size = max(img_w, img_h)
        new_w = int(sim_size * (img_w / max_size))
        new_h = int(sim_size * (img_h / max_size))
        paste_x = sim_size // 2 - new_w // 2
        paste_y = sim_size // 2 - new_h // 2
        img = img.resize((new_w, new_h))
        final_img = Image.new("RGBA", (sim_size, sim_size))
        final_img.paste(img, (paste_x, paste_y))
        final_img_arr = np.array(final_img).transpose((2, 0, 1)) / 255
        alpha_mask = (
            (final_img_arr[3] > 0.5).reshape((1, sim_size, sim_size)).astype(float)
        )
        final_img_arr = final_img_arr * alpha_mask
        final_img_arr[3] = alpha_mask
        final_img_arrs.append(final_img_arr)
    final_img_arrs = np.stack(final_img_arrs)

    # Set up initial seed image
    seed_arr = np.zeros((state_size, sim_size, sim_size))
    seed_arr[3:, sim_size // 2, sim_size // 2] = 1.0

    if args.eval:
        net = UpdateNet(state_size, dna_size).to(device)
        enc_net = EncoderNet(dna_size).to(device)
        net.load_state_dict(torch.load("temp/net.pt"))
        enc_net.load_state_dict(torch.load("temp/enc_net.pt"))

        with torch.no_grad():
            # Single image
            if args.t1 is None:
                ax = plt.subplot()
                final = (
                    torch.from_numpy(final_img_arrs[int(args.pic)])
                    .unsqueeze(0)
                    .to(device)
                )
                ax.set_facecolor("gray")
                ax.imshow(final_img_arrs[int(args.pic)][:4].transpose(1, 2, 0))
                plt.show()
                seed = torch.zeros((1, state_size), device=device)
                seed[:, 3:] = 1.0
                dna = (
                    enc_net(final)
                    .reshape(1, dna_size, 1, 1)
                    .repeat(1, 1, sim_size, sim_size)
                )  # Shape: (1, dna_size, sim_size, sim_size)
                curr_state = torch.zeros(
                    (1, sim_size, sim_size, state_size), device=device
                )
                curr_state[:, sim_size // 2, sim_size // 2] = seed
                curr_state = curr_state.permute(
                    0, 3, 1, 2
                )  # Shape: (1, state_size, sim_size, sim_size)
                for i in tqdm(range(max_train)):
                    curr_state = perform_update(curr_state, dna, net, min_a_alive, sim_size, cell_update, device)
                    ax = plt.subplot()
                    ax.set_facecolor("gray")
                    ax.imshow(
                        np.clip(
                            curr_state[0][:4].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0
                        )
                    )
                    plt.savefig(f"temp/generated/dna_trans/{i}.png")

            # Transitioning
            else:
                t1 = int(args.t1)
                t2 = int(args.t2)
                ax = plt.subplot()
                final1 = torch.from_numpy(final_img_arrs[t1]).unsqueeze(0).to(device)
                final2 = torch.from_numpy(final_img_arrs[t2]).unsqueeze(0).to(device)
                ax.set_facecolor("gray")
                # ax.imshow(final_img_arrs[t1][:4].transpose(1, 2, 0))
                # plt.show()
                seed = torch.zeros((1, state_size), device=device)
                seed[:, 3:] = 1.0
                dna1 = (
                    enc_net(final1)
                    .reshape(1, dna_size, 1, 1)
                    .repeat(1, 1, sim_size, sim_size)
                )  # Shape: (1, dna_size, sim_size, sim_size)
                dna2 = (
                    enc_net(final2)
                    .reshape(1, dna_size, 1, 1)
                    .repeat(1, 1, sim_size, sim_size)
                )
                curr_state = torch.zeros(
                    (1, sim_size, sim_size, state_size), device=device
                )
                curr_state[:, sim_size // 2, sim_size // 2] = seed
                curr_state = curr_state.permute(
                    0, 3, 1, 2
                )  # Shape: (1, state_size, sim_size, sim_size)
                print("Generating config 1...")
                for i in tqdm(range(max_train)):
                    curr_state = perform_update(curr_state, dna1, net, min_a_alive, sim_size, cell_update, device)
                    ax = plt.subplot()
                    ax.set_facecolor("gray")
                    ax.imshow(
                        np.clip(
                            curr_state[0][:4].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0
                        )
                    )
                    plt.savefig(f"temp/generated/dna_trans/{i}.png")

                print("Transitioning to config 2...")
                for i in tqdm(range(max_train)):
                    curr_state = perform_update(curr_state, dna2, net, min_a_alive, sim_size, cell_update, device)
                    ax = plt.subplot()
                    ax.set_facecolor("gray")
                    ax.imshow(
                        np.clip(
                            curr_state[0][:4].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0
                        )
                    )
                    plt.savefig(f"temp/generated/dna_trans/{max_train + i}.png")
        quit()

    # Train
    net = UpdateNet(state_size, dna_size).to(device)
    enc_net = EncoderNet(dna_size).to(device)
    opt = torch.optim.Adam(
        list(enc_net.parameters()) + list(net.parameters()), lr=0.001
    )
    pool = np.stack([seed_arr for _ in range(pool_size)], 0)
    pool_losses = np.zeros((pool_size,))
    for i in tqdm(range(train_iters)):
        # Sample a batch from the pool
        idxs = list(range(pool_size))
        random.shuffle(idxs)
        pool_idxs = np.array(idxs[:batch_size])
        curr_state = pool[pool_idxs]

        # Replace the sample with the highest loss with the seed state
        highest_loss = np.argmax(pool_losses[pool_idxs], 0)
        curr_state[highest_loss] = seed_arr

        img_idxs = list(range(final_img_arrs.shape[0]))
        random.shuffle(img_idxs)
        img_idxs = img_idxs[:batch_size]
        final = torch.from_numpy(final_img_arrs[np.array(img_idxs)]).to(device)
        dna = (
            enc_net(final)
            .reshape(batch_size, dna_size, 1, 1)
            .repeat(1, 1, sim_size, sim_size)
        )  # Shape: (batch_size, dna_size, sim_size, sim_size)
        curr_state = torch.from_numpy(curr_state).to(
            device
        )  # Shape: (batch_size, state_size, sim_size, sim_size)
        for _ in range(random.randrange(min_train, max_train)):
            curr_state = perform_update(curr_state, dna, net, min_a_alive, sim_size, cell_update, device)

        cmp_state = curr_state[:, :4, :, :]
        opt.zero_grad()
        loss = ((cmp_state - final) ** 2).mean([1, 2, 3])
        loss.mean().backward()
        tqdm.write(f"Loss: {loss.mean().item()}")
        opt.step()

        # Add back outputs to pool
        pool[pool_idxs] = curr_state.detach().cpu().numpy()
        pool_losses[pool_idxs] = loss.detach().cpu().numpy()

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
