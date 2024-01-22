"""
This experiment uses a chunked NCA approach on a 3d world.
"""
import open3d as o3d  # type: ignore
import numpy as np

# o3d.visualization.draw_geometries([], width=400, height=400, top=400) # Have to render a window first for Open3D output

import argparse
import random
from PIL import Image  # type: ignore
import torch
from torch import nn
from matplotlib import pyplot as plt  # type: ignore
from tqdm import tqdm
import torch.utils.checkpoint

# Params
# model_path = "temp/stanford-bunny.obj"
min_train = 16  # Minimum number of training steps.
max_train = 32  # Maximum number of training steps.
state_size = 16  # Number of states in a cell, including RGBA.
cell_update = 0.5  # Probability of a cell being updated.
train_iters = 100000  # Number of total training iterations.
min_a_alive = 0.1  # To be considered alive, a cell in the neighborhood's alpha value must be at least this.
pool_size = 64  # Number of items in the pool.
batch_size = 4  # Number of items per batch.
chunk_size = 32  # Size of each chunk. This is the size of the input passed to the net.
chunks_per_side = 1  # Number of chunks per side.
segment_size = 4  # Number of times the sim will run before gradient checkpointing.
device = torch.device("cuda")


class UpdateNet(nn.Module):
    """
    Given the current sim state, returns the update rule for each cell.
    """

    def __init__(self, state_size: int):
        super().__init__()
        out_conv = nn.Conv3d(128, state_size, 1, bias=False)
        out_conv.weight = nn.Parameter(torch.zeros(out_conv.weight.shape))
        self.net = nn.Sequential(
            nn.Conv3d(state_size * 4, 128, 1, bias=False),
            nn.ReLU(),
            out_conv,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.net(x)
        return update


def perform_update(
    curr_state: torch.Tensor,  # Shape: (batch_size, state_size, sim_size, sim_size, sim_size)
    net: nn.Module,
    sobel_x: torch.Tensor,
    sobel_y: torch.Tensor,
    sobel_z: torch.Tensor,
) -> torch.Tensor:
    """
    Performs one update step, returning the next state.
    """
    batch_size = curr_state.shape[0]
    state_size = curr_state.shape[1]

    # Compute perception vectors
    padding_size = 2  # We need to look 2 cells out to accurately compute chunk interactions and alive masking
    new_curr_state = torch.zeros(curr_state.shape)
    for z in range(chunks_per_side):
        for y in range(chunks_per_side):
            for x in range(chunks_per_side):
                z_min = max(chunk_size * z - padding_size, 0)
                z_max = min(
                    chunk_size * (z + 1) + padding_size, chunks_per_side * chunk_size
                )
                c_z_min = chunk_size * z
                c_z_max = chunk_size * (z + 1)

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
                curr_state_chunk = curr_state[
                    :,
                    :,
                    z_min:z_max,
                    y_min:y_max,
                    x_min:x_max,
                ]

                # Skip if cell is empty
                if curr_state_chunk.count_nonzero() == 0:
                    continue

                curr_state_chunk = curr_state_chunk.float().to(device)

                max_a = torch.max_pool3d(
                    curr_state_chunk[:, 0, :, :, :], 3, padding=1, stride=1
                ).unsqueeze(
                    1
                )  # Size: (batch_size, 1, chunk_size, chunk_size, chunk_size)
                mask_a = max_a > min_a_alive

                grad_x = nn.functional.conv3d(
                    curr_state_chunk, sobel_x, groups=state_size, padding=1
                )
                grad_y = nn.functional.conv3d(
                    curr_state_chunk, sobel_y, groups=state_size, padding=1
                )
                grad_z = nn.functional.conv3d(
                    curr_state_chunk, sobel_z, groups=state_size, padding=1
                )
                net_inpt = torch.concatenate(
                    [curr_state_chunk, grad_x, grad_y, grad_z], 1
                )
                update = net(net_inpt)
                update_mask = (
                    torch.distributions.Uniform(0.0, 1.0)
                    .sample(
                        torch.Size(
                            (
                                batch_size,
                                (z_max - z_min),
                                (y_max - y_min),
                                (x_max - x_min),
                            )
                        )
                    )
                    .unsqueeze(1)
                ).to(
                    device
                ) < cell_update  # Shape: (batch_size, 1, chunk_size + padding, chunk_size + padding, chunk_size + padding)
                curr_state_chunk = curr_state_chunk + update * update_mask
                curr_state_chunk = curr_state_chunk * mask_a
                new_curr_state[
                    :, :, c_z_min:c_z_max, c_y_min:c_y_max, c_x_min:c_x_max
                ] = curr_state_chunk.cpu()[
                    :,
                    :,
                    (c_z_min - z_min) : (c_z_min - z_min + chunk_size),
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
        sobel_z: torch.Tensor,
        dummy: torch.Tensor,
    ) -> torch.Tensor:
        for _ in range(segment_size):
            curr_state = perform_update(curr_state, net, sobel_x, sobel_y, sobel_z)
        return curr_state

    return run_forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_single", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Load model
    sim_size = chunk_size * chunks_per_side
    bunny = o3d.data.ArmadilloMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.scale(
        1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
        center=mesh.get_center(),
    )
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh, voxel_size=1.1 / sim_size
    )
    final_voxel_arr = np.zeros((1, sim_size, sim_size, sim_size), dtype=bool)
    for v in voxel_grid.get_voxels():
        final_voxel_arr[0, v.grid_index[2], v.grid_index[1], v.grid_index[0]] = True
    seed_point = (sim_size // 2, sim_size // 2, sim_size // 2 - 2)
    next_voxels = [seed_point]
    seen_voxels = set()
    while len(next_voxels) > 0:
        next_voxel = next_voxels.pop(0)
        final_voxel_arr[0, next_voxel[0], next_voxel[1], next_voxel[2]] = True
        for z in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    if (z == 0 and y == 0 and x == 0) or (abs(x) + abs(y) + abs(z)) > 1:
                        continue
                    new_vox = (
                        min(max(next_voxel[0] + z, 0), sim_size - 1),
                        min(max(next_voxel[1] + x, 0), sim_size - 1),
                        min(max(next_voxel[2] + y, 0), sim_size - 1),
                    )
                    if (
                        new_vox not in seen_voxels
                        and not final_voxel_arr[0, new_vox[0], new_vox[1], new_vox[2]]
                    ):
                        next_voxels.append(new_vox)
                        seen_voxels.add(new_vox)
    final_voxel_arr = final_voxel_arr.transpose(0, 3, 1, 2).astype(float)
    # ax = plt.figure().add_subplot(projection="3d")
    # ax.voxels(final_voxel_arr[0])
    # plt.show()
    # plt.imshow(final_voxel_arr[0][sim_size // 2])
    # plt.show()

    # Set up initial seed voxel
    seed_arr = np.zeros((state_size, sim_size, sim_size, sim_size))
    seed_arr[0, seed_point[0], seed_point[1], seed_point[2]] = 1.0

    # Filters
    sobel_x_ = torch.tensor(
        [
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        ],
        dtype=torch.float,
    ).to(device)
    sobel_x = sobel_x_.unsqueeze(0).unsqueeze(0).repeat(state_size, 1, 1, 1, 1)
    sobel_y = (
        sobel_x_.swapaxes(0, 2).unsqueeze(0).unsqueeze(0).repeat(state_size, 1, 1, 1, 1)
    )
    sobel_z = (
        sobel_x_.swapaxes(0, 1).unsqueeze(0).unsqueeze(0).repeat(state_size, 1, 1, 1, 1)
    )

    if args.eval or args.eval_single:
        net = UpdateNet(state_size).to(device)
        net.load_state_dict(torch.load("temp/net.pt"))

        with torch.no_grad():
            curr_state = torch.from_numpy(seed_arr).unsqueeze(0)
            for i in tqdm(range(1000)):
                curr_state = perform_update(curr_state, net, sobel_x, sobel_y, sobel_z)
                if i % 10 == 0 and not args.eval_single:
                    ax = plt.figure().add_subplot(projection="3d")
                    # ax.voxels(torch.argmax(curr_state[0, :2], dim=0) == 1)
                    ax.voxels(curr_state[0, 0] > min_a_alive)
                    plt.savefig(f"temp/generated/chunks_3d/{i}.png")
                    plt.close()
            
            if args.eval_single:
                ax = plt.figure().add_subplot(projection="3d")
                ax.voxels(curr_state[0, 0] > min_a_alive)
                plt.savefig(f"temp/generated/chunks_3d/single.png")
                plt.close()
        quit()

    # Train
    net = UpdateNet(state_size).to(device)
    if args.resume:
        net.load_state_dict(torch.load("temp/net.pt"))
    opt = torch.optim.Adam(net.parameters(), lr=0.00001)
    pool = np.stack([seed_arr for _ in range(pool_size)], 0)
    pool_losses = np.zeros((pool_size,))
    last_index = 0
    for i in tqdm(range(train_iters)):
        # Set batch size to 1 at beginning to encourage rules to start being applied
        if i < 20 and not args.resume:
            iter_batch_size = 1
        else:
            iter_batch_size = batch_size

        if i < 1 and not args.resume:
            final = torch.ones((iter_batch_size, 1, sim_size, sim_size, sim_size))
        else:
            final = torch.from_numpy(final_voxel_arr).repeat(
                iter_batch_size, 1, 1, 1, 1
            )

        # Sample a batch from the pool
        idxs = list(range(pool_size))
        random.shuffle(idxs)
        pool_idxs = np.array(idxs[:iter_batch_size])
        curr_state = pool[pool_idxs]

        # Replace the sample with the highest loss with the seed state.
        highest_loss = np.argmax(pool_losses[pool_idxs], 0)
        curr_state[highest_loss] = seed_arr

        # If any samples are empty, replace those too.
        for i in range(iter_batch_size):
            if not curr_state[i].any():
                curr_state[i] = seed_arr
        
        # Perform training
        curr_state = torch.from_numpy(
            curr_state
        )  # Shape: (batch_size, state_size, sim_size, sim_size, sim_size)
        for _ in tqdm(range(random.randrange(min_train, max_train) // segment_size)):
            # try:
            curr_state = torch.utils.checkpoint.checkpoint(
                gen_forward(segment_size),
                curr_state.clone(),
                net,
                sobel_x,
                sobel_y,
                sobel_z,
                torch.ones([1], requires_grad=True),
            )
            # except KeyboardInterrupt:
            #     quit()
            # except Exception as e:
            #     print(e)

        opt.zero_grad()
        total_losses = torch.zeros([iter_batch_size]).to(device)
        for z in range(chunks_per_side):
            for y in range(chunks_per_side):
                for x in range(chunks_per_side):
                    cmp_state = curr_state[
                        :,
                        :1,
                        (chunk_size * z) : (chunk_size * (z + 1)),
                        (chunk_size * y) : (chunk_size * (y + 1)),
                        (chunk_size * x) : (chunk_size * (x + 1)),
                    ]

                    # Skip empty cells (gradient should be 0 here?)
                    if cmp_state.count_nonzero() == 0:
                        continue

                    cmp_state = cmp_state.to(device)
                    final_cmp = (
                        final[
                            :,
                            :1,
                            (chunk_size * z) : (chunk_size * (z + 1)),
                            (chunk_size * y) : (chunk_size * (y + 1)),
                            (chunk_size * x) : (chunk_size * (x + 1)),
                        ]
                        .to(device)
                        .float()
                    )
                    loss = ((torch.sigmoid(cmp_state) - final_cmp) ** 2).mean(
                        [1, 2, 3, 4]
                    )
                    total_losses += loss
        total_losses.mean().backward()
        opt.step()
        tqdm.write(f"Loss: {total_losses.mean().item()}")

        # Add back outputs to pool
        new_idxs = (last_index + np.arange(0, iter_batch_size)) % pool_size
        pool[new_idxs] = curr_state.detach().cpu().numpy()
        pool_losses[new_idxs] = total_losses.detach().cpu().numpy()
        last_index = (last_index + iter_batch_size) % pool_size

        # Save network
        if (i + 1) % 4 == 0:
            torch.save(net.cpu().state_dict(), "temp/net.pt")
            net.to(device)


if __name__ == "__main__":
    main()
