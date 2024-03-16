import torch.nn as nn
import sys
import logging
import os
# from chop.models.vision.vgg_cifar import get_vgg7
from pathlib import Path
from pprint import pprint as pp
import torch

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity

from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph
from chop.models import get_model_info, get_model

set_logging_verbosity("info")


# Model and dataset names
batch_size = 8
model_name = "vgg7"
dataset_name = "cifar10"

# Define VGG7 Model
class VGG7(nn.Module):
    def __init__(self, image_size: list[int], num_classes: int) -> None:
        super().__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(image_size[0], 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * (image_size[1] // 8) * (image_size[2] // 8), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.last_layer = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.last_layer(x)
        return x


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

# checkpoint
CHECKPOINT_PATH = "/mnt/c/users/KelseyJing/mase/machop/test-accu-0.9332.ckpt"

model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)
model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)


input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)
