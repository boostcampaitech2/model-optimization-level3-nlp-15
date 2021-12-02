"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
from datetime import datetime
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
from torchvision import models
import timm
import glob

DEBUG = True


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

    # Load model
    model_architecture = "resnet18"
    model = models.resnet18(pretrained=True)

    # replace fully connected layers
    num_in_features = model.fc.in_features
    model.fc = nn.Linear(num_in_features, 6)

    # check the layer info
    print("======original model======")
    print(model_info(model))
    for child in model.children():
        print(child)

    # remove last two conv layers from the model
    # deep copy the model
    newmodel = model
    # print last BasicBlock from the model
    # print("======last BasicBlock======")
    # print(list(newmodel.children())[-1])
    # print modules
    print("======modules======")
    print(newmodel.modules)

    # remove layer 3 and 4 from the model
    del newmodel.layer3
    del newmodel.layer4

    # replace fully connected layers
    num_in_features = 128
    newmodel.fc = nn.Linear(num_in_features, 6)

    print("======changed model======")
    print(model_info(newmodel))
    for child in newmodel.children():
        print(child)

    # move model to device
    newmodel.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)
    model_path = os.path.join(log_dir, f"best_teacher_{model_architecture}.pt")
    # Create optimizer, scheduler, criterion
    optimizer = SGDP(model.parameters(), lr=data_config["INIT_LR"], momentum=0.9)
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
        model=model,
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
    # model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")

    parser.add_argument("--data", default="configs/data/taco.yaml", type=str, help="data config")
    args = parser.parse_args()

    data_config = read_yaml(cfg=args.data)
    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("teacher", "latest"))
    if os.path.exists(log_dir) and DEBUG == False:
        # find *.pt file in log_dir
        previous_model_path = glob.glob(os.path.join(log_dir, "*.pt"))[0]
        modified = datetime.fromtimestamp(os.path.getmtime(previous_model_path))
        new_log_dir = os.path.dirname(log_dir) + "/" + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        model_config=None,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )
