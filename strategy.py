from typing import Union, Tuple
from itertools import product, chain
import heapq
import numpy as np
from skimage.metrics import structural_similarity
from matplotlib import pyplot as plt
from pathlib import Path

import algorithm
from PIL import Image


class Position:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def move(self, step):
        return Position(self.row + step.row, self.col + step.col)

    def flatten(self):
        return self.row, self.col

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __repr__(self):
        return f"POS_({self.row}, {self.col})"

    def __lt__(self, other):
        return self.row < other.row and self.col < other.col


class Direction:
    UP = Position(-1, 0)
    DOWN = Position(1, 0)
    LEFT = Position(0, -1)
    RIGHT = Position(0, 1)
    OPTIONS = (UP, DOWN, LEFT, RIGHT)


def check_reachability(padded_map, start_pos: Position, end_pos: Position):
    def check_pos_in_map(pos):
        return 0 <= pos.row < max_row and 0 <= pos.col < max_col

    max_row, max_col = padded_map.shape

    def dfs(current_pos: Position, turns: int, last_step: Union[Position, None]):

        if current_pos == end_pos:
            return check_pos_in_map(current_pos)

        for step in Direction.OPTIONS:
            new_pos = current_pos.move(step)
            new_turns = turns + 1 if last_step is not None and step != last_step else turns
            if new_turns > 2:
                continue

            if check_pos_in_map(new_pos) and (padded_map[new_pos.row, new_pos.col] or new_pos == end_pos):
                if dfs(new_pos, new_turns, step):
                    return True

        return False

    return dfs(start_pos, 0, None)


def check_not_background(padded_bg_mask, pos: Position):
    return not padded_bg_mask[pos.row, pos.col]


class BaseAgent:
    def __init__(self, max_pairs=100):
        self.max_pairs = max_pairs

    def get_patches_sim(self, patches, indices_pairs: Tuple[Tuple[int, int]]):
        return np.random.rand(len(indices_pairs))

    def generate_strategy(self, padded_bg_mask, patches, verbose=False):
        rows, cols = patches.shape[:2]
        padded_step = Position(1, 1)
        patches_idx = list(product(range(rows), range(cols)))

        indices_pairs = []
        position_pairs = []

        for start_idx, end_idx in product(patches_idx, patches_idx):
            if start_idx > end_idx:
                # check reachability; index changes after padding
                start_pos, end_pos = Position(*start_idx).move(padded_step), \
                                     Position(*end_idx).move(padded_step)
                if check_reachability(padded_bg_mask, start_pos, end_pos):
                    indices_pairs.append((start_idx, end_idx))
                    position_pairs.append((start_pos, end_pos))

        pairs_sim = self.get_patches_sim(patches, indices_pairs)

        pairs = []
        for sim, pos_pair in zip(pairs_sim, position_pairs):
            start_pos, end_pos = pos_pair
            heapq.heappush(pairs, (-sim, start_pos, end_pos))

        # then sort and check whether the position is a hole
        # the bg_mask is dynamically filled, assuming all steps are valid

        filtered_pairs = []
        for _ in range(min(self.max_pairs, len(pairs))):
            neg_sim, start_pos, end_pos = heapq.heappop(pairs)
            if check_not_background(padded_bg_mask, start_pos) and check_not_background(padded_bg_mask, end_pos):
                filtered_pairs.append((neg_sim, start_pos, end_pos))
                padded_bg_mask[start_pos.flatten()] = 1
                padded_bg_mask[end_pos.flatten()] = 1

        pairs = tuple(filtered_pairs)

        if verbose:
            print("Top 10 pairs with highest similarity: ")
            print("Sim\tStart_pos\tEnd_pos")
            fig, axes = plt.subplots(ncols=2, nrows=10)
            for i, (neg_sim, start_pos, end_pos) in enumerate(pairs[:10]):
                print(f"{-neg_sim:.6f}\t{start_pos}\t{end_pos}")
                axes[i, 0].axis("off")
                axes[i, 1].axis("off")
                axes[i, 0].imshow(patches[start_pos.row-1, start_pos.col-1])
                axes[i, 1].imshow(patches[end_pos.row-1, end_pos.col-1])
            plt.tight_layout()

        strategy = []
        for sim, start_pos, end_pos in pairs:
            # make it circular to avoid mis-click
            strategy.append((start_pos.flatten(), end_pos.flatten(), start_pos.flatten()))

        return tuple(strategy)


class EdgeDetectionAgent(BaseAgent):
    def __init__(self, max_pairs=100):
        super().__init__(max_pairs)

    def get_patches_sim(self, patches, indices_pairs):

        edges = {idx: algorithm.detect_edges(patches[idx]) for idx in set(chain.from_iterable(indices_pairs))}

        sims = []
        for start_idx, end_idx in indices_pairs:
            start_edges = edges[start_idx]
            end_edges = edges[end_idx]
            sims.append(structural_similarity(start_edges, end_edges))
        return np.array(sims)


# class SiameseNetAgent(BaseAgent):
#     def __init__(self, max_pairs=100):
#         super().__init__(max_pairs)
#
#     def get_patches_sim(self, patches, indices_pairs):
#         pairs = []
#         for start_idx, end_idx in indices_pairs:
#             start_patch, end_patch = patches[start_idx], patches[end_idx]
#             patch_size = start_patch.shape[0]
#             pair = Image.new("RGB", size=(patch_size*2, patch_size))
#             pair.paste(Image.fromarray(start_patch), (0, 0, patch_size, patch_size))
#             pair.paste(Image.fromarray(start_patch), (patch_size, 0, 2*patch_size, patch_size))
#             pairs.append(pair)
#         pairs = list(map(model.compose_transform(), pairs))
#         dataset = model.construct_dataset_from_pairs(pairs)
#         net = model.SiameseNet.load_from_checkpoint(Path(__file__).parent / "checkpoints/aug_model.ckpt")
#         return model.SiameseNet.infer_on_dataset(net, dataset, thresh=None)
