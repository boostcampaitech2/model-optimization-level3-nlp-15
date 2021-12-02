"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
from pickle import TRUE
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from collections import OrderedDict

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info
from src.modules.mbconv import MBConvGenerator
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from adamp import SGDP

DEBUG = TRUE

# https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    state_dict = load_state_dict_from_url(model_urls["resnet18"])
    model_state_dict = model_instance.model.state_dict()
    model_state_keys = [i for i in model_instance.model.state_dict()]
    print(f"Model state keys: {model_state_keys}")
    state_dict_keys = [i for i in state_dict]
    new_state_dict = OrderedDict()

    for key in state_dict_keys:
        print(key)

    # Pretrained weights
    for i in range(len(model_state_keys) - 2):
        model_state = model_state_keys[i]
        pretrained_weight_state = state_dict_keys[i]
        new_state_dict[model_state] = state_dict[pretrained_weight_state]

    # Classifier
    for i in range(len(model_state_keys) - 2, len(model_state_keys)):
        model_state = model_state_keys[i]
        new_state_dict[model_state] = model_state_dict[model_state_keys[i]]

    model_instance.model.load_state_dict(new_state_dict, strict=False)

    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    optimizer = SGDP(model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config["INIT_LR"],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None

    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/resnet18.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument("--data", default="configs/data/taco.yaml", type=str, help="data config")
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", "latest"))

    if os.path.exists(log_dir) and DEBUG == False:
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + "/best.pt"))
        new_log_dir = os.path.dirname(log_dir) + "/" + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=True,
        device=device,
    )
