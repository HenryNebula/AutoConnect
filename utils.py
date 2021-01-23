import io
from datetime import datetime
from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import TensorDataset
import numpy as np

from ipywidgets import widgets, Output
from IPython.display import display, clear_output
from PIL import Image, ImageDraw


def format_current_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format="png")
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def create_checkbox(value):
    checkbox = widgets.Checkbox(value=value,
                                description='Correct',
                                disabled=False,
                                indent=False,
                                layout=widgets.Layout(width='100px', height='50px', margin='auto'))
    return checkbox


def tensor_to_image(tensor: torch.Tensor) -> Image:
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    img = inv_normalize(tensor)
    img = np.transpose(img.numpy(), axes=[1, 2, 0])
    img = np.clip(img, 0, 1) * 255
    img = Image.fromarray(np.array(img, dtype=np.uint8))
    return img


def create_expanded_button(description, button_style):
    return widgets.Button(description=description,
                          button_style=button_style,
                          layout=widgets.Layout(height='auto', width='auto'))


class ValidateWidget:

    def __init__(self, dataset, num_instances=10):
        self.dataset = dataset
        self.num_instances = num_instances
        self.corrected_mask = []

        self.checkboxes = [create_checkbox(True) for _ in range(num_instances)]
        self.next_button = create_expanded_button("Next Round", "success")
        self.next_button.on_click(callback=self.on_next_button_clicked)
        self.finish_button = create_expanded_button("Finish", "danger")
        self.finish_button.on_click(callback=self.on_finish_button_clicked)
        self.grid = widgets.GridspecLayout(3, num_instances)
        for _ in range(num_instances):
            self.grid[1, _] = self.checkboxes[_]
        self.grid[2, :5] = self.next_button
        self.grid[2, 5:] = self.finish_button

    def get_image(self, idx):
        pair, label = self.dataset[idx]
        pair = tensor_to_image(pair)
        pair.thumbnail((39 * 2, 39))

        if not label:
            draw = ImageDraw.Draw(pair)
            draw.line((0, 0) + pair.size, fill=128, )
            draw.line((0, pair.size[1], pair.size[0], 0), fill=128)

        return image_to_byte_array(pair)

    def display(self):

        if self.grid is None:
            print("The grid has been removed. Try initiate this object again.")
            return

        start_idx = len(self.corrected_mask)
        if start_idx >= len(self.dataset):
            correct_num = np.sum(self.corrected_mask)
            self.on_finish_button_clicked(widget=self.finish_button)
            print(f"Finished validating the whole dataset. {correct_num} / {len(self.corrected_mask)} are correct.")
        else:
            for wid in range(10):
                self.grid[0, wid] = widgets.Image(value=self.get_image(start_idx + wid),
                                                  format="png",
                                                  indent=False)
                self.checkboxes[wid].value = True
            clear_output()
            display(self.grid)

    def on_next_button_clicked(self, widget):
        print("next round button clicked.")
        self.corrected_mask.extend([b.value for b in self.checkboxes])
        self.display()

    def on_finish_button_clicked(self, widget):
        clear_output()
        self.grid = None

    def get_corrected_dataset(self):
        pairs = []
        new_labels = []

        correct_num = np.sum(self.corrected_mask)

        if len(self.corrected_mask) < len(self.dataset):
            self.corrected_mask.extend([True] * (len(self.dataset) - len(self.corrected_mask)))

        for (pair, label), mask in zip(self.dataset, self.corrected_mask):
            pairs.append(pair)
            new_labels.append(label if mask else 1. - label)

        print(f"{correct_num} / {len(self.corrected_mask)} are correct.")

        return TensorDataset(torch.stack(pairs, dim=0), torch.Tensor(new_labels))

    @staticmethod
    def save_dataset(dataset, save_dir, val_ratio=0.2, save_incorrect_only=True):
        split_rnd = np.random.rand(len(dataset))
        train_dir = Path(save_dir) / Path("train/")
        val_dir = Path(save_dir) / Path("val/")
        cnt = 0
        for (pair, label), rnd in zip(dataset, split_rnd):
            if label and save_incorrect_only:
                continue
            dir_ = val_dir if rnd < val_ratio else train_dir
            ts = format_current_timestamp() + f"_id_{cnt}"
            path = dir_ / Path(f"pos/{ts}.png") if label else dir_ / Path(f"neg/{ts}.png")
            img = tensor_to_image(pair)
            img.save(path)
            cnt += 1
