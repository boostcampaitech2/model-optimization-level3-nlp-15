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

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info
from adamp import SGDP, AdamP
import timm
from transformers import AdamW

class CustomModel(nn.Module):
    def __init__(self, num_garbage_classes: int = 6):
        super(CustomModel, self).__init__()
        # Transfer learning to add final layers in the end.
        # Model Comaprison: https://paperswithcode.com/sota/image-classification-on-imagenet
        # self.backbone = models.resnet50(pretrained=True)
        # self.backbone = models.resnext50_32x4d(pretrained=True)
        # self.backbone.fc = nn.Linear(in_features=2048, out_features=18, bias=True)


        # ViT Model Explanation: https://huggingface.co/google/vit-base-patch16-384
        # Pretrained Model SourcE: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # 예시) vit_large_patch32_224_in21k
        # - model size: tiny, small, base, large | 모델 크기
        # - fixed-size patches: 16, 32               | 이미지가 몇 개로 쪼개져서 들어가는지
        # - in21k: imagenet 21000 classes       | Pretrained Imagenet Dataset의 클래스 개수

        self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
        
        # replacing the last Linear layer with the defined problem's number of classes
        num_input_features = self.model.head.in_features # pretrained model's default fully connected Linear Layer
        self.model.head = nn.Linear(in_features=num_input_features, out_features=num_garbage_classes, bias=True)  # replacing output with 18

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

def train(
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    timm_model = CustomModel()
    model_path = os.path.join(log_dir, "best_teacher.pt")
    print(f"Model save path: {model_path}")
    

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    optimizer = AdamW(
        timm_model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    )
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
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=timm_model,
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
    test_loss, test_f1, test_acc = trainer.test(
        model=timm_model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/mobilenetv3.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))

    if os.path.exists(log_dir): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    test_loss, test_f1, test_acc = train(
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )

