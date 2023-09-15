"""
This experiment scales up NCA to operate on chunks in a world while preserving
gradients. This allows it to scale.
"""
import argparse
import random
from PIL import Image  # type: ignore
import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm
import torch.utils.checkpoint

# Params
img_path = "temp/cat.png"
min_train = 64  # Minimum number of training steps.
max_train = 96  # Maximum number of training steps.
state_size = 16  # Number of states in a cell, including RGBA.
cell_update = 0.5  # Probability of a cell being updated.
train_iters = 4000  # Number of total training iterations.
min_a_alive = 0.1  # To be considered alive, a cell in the neighborhood's alpha value must be at least this.
pool_size = 1024  # Number of items in the pool.
batch_size = 4  # Number of items per batch.
chunk_size = 64  # Size of each chunk. This is the size of the input passed to the net.
chunks_per_side = 4  # Number of chunks per side.
segment_size = 8  # Number of times the sim will run before gradient checkpointing.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.net(x)
        return update


def perform_update(
    curr_state: torch.Tensor,  # Shape: (batch_size, state_size, sim_size, sim_size)
    net: nn.Module,
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
) -> torch.Tensor:
    """
    Performs one update step, returning the next state.
    """
    batch_size = curr_state.shape[0]
    state_size = curr_state.shape[1]

    # Compute perception vectors
    padding_size = 2  # We need to look 2 cells out to accurately compute chunk interactions and alive masking
    new_curr_state = torch.zeros(curr_state.shape)
    for y in range(chunks_per_side):
        for x in range(chunks_per_side):
            y_min = max(chunk_size * y - padding_size, 0)
            y_max = min(
                chunk_size * (y + 1) + padding_size, chunks_per_side * chunk_size
            )
            c_y_min = chunk_size * y
            c_y_max = chunk_size * (y + 1)
            x_min = max(chunk_size * x - padding_size, 0)
            x_max = min(
                chunk_size * (x + 1) + padding_size, chunks_per_side * chunk_size
            )
            c_x_min = chunk_size * x
            c_x_max = chunk_size * (x + 1)
            curr_state_chunk = (
                curr_state[
                    :,
                    :,
                    y_min:y_max,
                    x_min:x_max,
                ]
            )

            # Skip if cell is empty
            if curr_state_chunk.count_nonzero() == 0:
                continue

            curr_state_chunk = curr_state_chunk.float().to(device)

            grad_x = nn.functional.conv2d(
                curr_state_chunk, sobel_x, groups=state_size, padding=1
            )
            grad_y = nn.functional.conv2d(
                curr_state_chunk, sobel_y, groups=state_size, padding=1
            )
            net_inpt = torch.concatenate([curr_state_chunk, grad_x, grad_y], 1)
            update = net(net_inpt)
            update_mask = (
                torch.distributions.Uniform(0.0, 1.0)
                .sample(torch.Size((batch_size, (y_max - y_min), (x_max - x_min))))
                .unsqueeze(1)
            ).to(
                device
            ) < cell_update  # Shape: (batch_size, 1, chunk_size + padding, chunk_size + padding)
            curr_state_chunk = curr_state_chunk + update * update_mask
            max_a = torch.max_pool2d(
                curr_state_chunk[:, 3, :, :], 3, padding=1, stride=1
            ).unsqueeze(
                1
            )  # Size: (batch_size, 1, chunk_size, chunk_size)
            mask_a = max_a > min_a_alive
            curr_state_chunk = curr_state_chunk * mask_a
            new_curr_state[
                :, :, c_y_min:c_y_max, c_x_min:c_x_max
            ] = curr_state_chunk.cpu()[
                :,
                :,
                (c_y_min - y_min) : (c_y_min - y_min + chunk_size),
                (c_x_min - x_min) : (c_x_min - x_min + chunk_size),
            ]
    return new_curr_state


def gen_forward(segment_size: int):
    """
    Creates a wrapper around the update function for gradient checkpointing.
    """

    def run_forward(
        curr_state: torch.Tensor,
        net: nn.Module,
        sobel_x: torch.Tensor,
        sobel_y: torch.Tensor,
        dummy: torch.Tensor,
    ) -> torch.Tensor:
        for _ in range(segment_size):
            curr_state = perform_update(curr_state, net, sobel_x, sobel_y)
        return curr_state

    return run_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    # Load image
    img = Image.open(img_path)
    img_w, img_h = img.size
    max_size = max(img_w, img_h)
    sim_size = chunk_size * chunks_per_side
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
    seed_arr[3:, sim_size // 2, sim_size // 2] = 1.0

    # Filters
    sobel_x_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float).to(
        device
    )
    sobel_x = sobel_x_.unsqueeze(0).unsqueeze(0).repeat(state_size, 1, 1, 1)
    sobel_y = sobel_x_.T.unsqueeze(0).unsqueeze(0).repeat(state_size, 1, 1, 1)

    if args.eval:
        net = UpdateNet(state_size).to(device)
        net.load_state_dict(torch.load("temp/net.pt"))

        with torch.no_grad():
            curr_state = torch.from_numpy(seed_arr).unsqueeze(0)
            for i in tqdm(range(200)):
                curr_state = perform_update(curr_state, net, sobel_x, sobel_y)
                plt.imshow(
                    np.clip(curr_state[0][:4].permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
                )
                plt.savefig(f"temp/generated/chunks/{i}.png")
                net.to(device)
        quit()

    # Train
    net = UpdateNet(state_size).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    pool = np.stack([seed_arr for _ in range(pool_size)], 0)
    pool_losses = np.zeros((pool_size,))
    for i in tqdm(range(train_iters)):
        # Set batch size to 1 at beginning to encourage rules to start being applied
        if i < 100:
            iter_batch_size = 1
        else:
            iter_batch_size = batch_size

        final = (
            torch.from_numpy(final_img_arr)
            .unsqueeze(0)
            .repeat(iter_batch_size, 1, 1, 1)
            .to(device)
        )

        # Sample a batch from the pool
        idxs = list(range(pool_size))
        random.shuffle(idxs)
        pool_idxs = np.array(idxs[:iter_batch_size])
        curr_state = pool[pool_idxs]

        # Replace the sample with the highest loss with the seed state
        highest_loss = np.argmax(pool_losses[pool_idxs], 0)
        curr_state[highest_loss] = seed_arr

        # Perform training
        curr_state = torch.from_numpy(
            curr_state
        )  # Shape: (batch_size, state_size, sim_size, sim_size)
        for _ in tqdm(range(random.randrange(min_train, max_train) // segment_size)):
            try:
                curr_state = torch.utils.checkpoint.checkpoint(
                    gen_forward(segment_size),
                    curr_state.clone(),
                    net,
                    sobel_x,
                    sobel_y,
                    torch.ones([1], requires_grad=True),
                )
            except KeyboardInterrupt:
                quit()
            except:
                print("No gradient. Ignoring...")

        opt.zero_grad()
        total_losses = torch.zeros([iter_batch_size]).to(device)
        for y in range(chunks_per_side):
            for x in range(chunks_per_side):
                cmp_state = curr_state[
                    :,
                    :4,
                    (chunk_size * y) : (chunk_size * (y + 1)),
                    (chunk_size * x) : (chunk_size * (x + 1)),
                ]

                cmp_state = cmp_state.to(device)
                final_cmp = final[
                    :,
                    :4,
                    (chunk_size * y) : (chunk_size * (y + 1)),
                    (chunk_size * x) : (chunk_size * (x + 1)),
                ].to(device)
                loss = ((cmp_state - final_cmp) ** 2).mean([1, 2, 3])
                total_losses += loss
        total_losses.mean().backward()
        opt.step()
        tqdm.write(f"Loss: {total_losses.mean().item()}")

        # Add back outputs to pool
        pool[pool_idxs] = curr_state.detach().cpu().numpy()
        pool_losses[pool_idxs] = total_losses.detach().cpu().numpy()

        # Save network
        if (i + 1) % 4 == 0:
            torch.save(net.cpu().state_dict(), "temp/net.pt")
            net.to(device)


if __name__ == "__main__":
    main()
