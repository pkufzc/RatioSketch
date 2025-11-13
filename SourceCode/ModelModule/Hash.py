"""Random hashing helper.

Produces deterministic random indices via a hash-derived seed.
"""

import torch
import torch.nn as nn
import xxhash


class RandomHash(nn.Module):
    """Returns random indices for given inputs using a hash-derived seed.

    This is used to simulate multiple hash functions for sketches.
    """

    def __init__(self, hash_seed, hash_range):
        super().__init__()
        self.hash_seed = hash_seed
        self.hash_range = hash_range

    # hash each sample by hash_num times
    def forward(self, input_tensor, sample_size, hash_num=1):
        with torch.no_grad():
            # derive a seed from the input content to make hashes repeatable
            seed = (
                xxhash.xxh32(
                    (str(input_tensor.sum().cpu().item()) + str(input_tensor.shape)),
                    seed=self.hash_seed,
                ).intdigest()
                % 1000
            )
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(input_tensor.device)
            generator.manual_seed(seed)
            hashed_indexes_tensor = torch.randint(
                low=0,
                high=self.hash_range,
                size=(sample_size, hash_num),
                requires_grad=False,
                generator=generator,
                device=input_tensor.device,
            )
            return hashed_indexes_tensor
