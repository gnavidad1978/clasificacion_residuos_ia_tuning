"""Modelos del proyecto: fabrica de redes y modulo de Lightning.

Redes disponibles (parametro ``backbone`` del YAML):
    mobilenet_v3_large  CNN ligera (torchvision)
    mobilenet_v3_small  CNN muy ligera (torchvision)
    resnet50            CNN estandar con conexiones residuales (torchvision)
    yolo11n_cls         CNN de la familia YOLO para clasificacion (Ultralytics)
    deit_small          Vision Transformer pequeno, de tamano comparable a ResNet-50 (timm)
    vit_b_16            Vision Transformer base (torchvision)

Todas parten de pesos preentrenados en ImageNet; solo se sustituye la ultima
capa por una de ``num_classes`` salidas y se reentrena la red completa.
"""

import torch
import torch.nn as nn
import lightning as L
import torchmetrics
from torchvision import models

from categorias import CATEGORIAS, MEDIA_IMAGENET, DESV_IMAGENET  # noqa: F401


class YOLOClasificador(nn.Module):
    """Cuerpo de YOLO11n-cls con una ultima capa nueva.

    Se entrena con el mismo bucle que las demas redes, no con el entrenador de
    Ultralytics, para que la unica diferencia entre modelos sea la arquitectura.
    YOLO se preentreno con fotos en escala 0 a 1 sin normalizacion de ImageNet,
    por eso aqui se deshace esa normalizacion antes de entrar a la red.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        from ultralytics import YOLO  # solo se importa si se usa este modelo

        base = YOLO("yolo11n-cls.pt" if pretrained else "yolo11n-cls.yaml").model
        capas = list(base.model)
        self.cuerpo = nn.Sequential(*capas[:-1])
        cabeza = capas[-1]
        self.conv, self.pool, self.drop = cabeza.conv, cabeza.pool, cabeza.drop
        self.linear = nn.Linear(cabeza.linear.in_features, num_classes)
        self.register_buffer("media", torch.tensor(MEDIA_IMAGENET).view(1, 3, 1, 1))
        self.register_buffer("desv", torch.tensor(DESV_IMAGENET).view(1, 3, 1, 1))
        for p in self.parameters():  # Ultralytics entrega los pesos congelados
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.desv + self.media
        x = self.cuerpo(x)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))


def crear_red(backbone: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Fabrica de modelos: devuelve la red pedida con su ultima capa ajustada."""
    if backbone == "mobilenet_v3_large":
        red = models.mobilenet_v3_large(weights="IMAGENET1K_V2" if pretrained else None)
        red.classifier[3] = nn.Linear(red.classifier[3].in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        red = models.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None)
        red.classifier[3] = nn.Linear(red.classifier[3].in_features, num_classes)
    elif backbone == "resnet50":
        red = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        red.fc = nn.Linear(red.fc.in_features, num_classes)
    elif backbone == "yolo11n_cls":
        red = YOLOClasificador(num_classes, pretrained)
    elif backbone == "deit_small":
        import timm  # solo se importa si se usa este modelo
        # Pesos de ImageNet-1K con la receta de DeiT (Touvron et al., 2021), igual que el ViT-B
        red = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=num_classes)
    elif backbone == "vit_b_16":
        red = models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        red.heads.head = nn.Linear(red.heads.head.in_features, num_classes)
    else:
        raise ValueError(
            f"backbone desconocido: '{backbone}'. Usa mobilenet_v3_large, mobilenet_v3_small, "
            "resnet50, yolo11n_cls, deit_small o vit_b_16."
        )
    return red


class ResiduosModule(L.LightningModule):
    """LightningModule para clasificar residuos en las categorias de la NAE-SEMADES-007/2008.

    Args:
        backbone:     Red a usar (ver la lista al inicio del archivo).
        num_classes:  Numero de categorias.
        pretrained:   Cargar pesos de ImageNet.
        optimizer:    "adamw", "adam", "radam" o "sgd".
        lr:           Tasa de aprendizaje.
        weight_decay: Decaimiento de pesos (regularizacion).
        momentum:     Momentum, solo para SGD.
        programador:  Como baja la tasa de aprendizaje.
                      "coseno": baja suavemente hasta max_epochs (receta original, 15 epocas).
                      "meseta": la reduce a la mitad cuando la perdida de validacion deja de
                      bajar; no depende de max_epochs, por eso se usa cuando no hay tope de epocas.
        paciencia_lr: Epocas sin mejora antes de reducir la tasa (solo "meseta").
    """

    def __init__(
        self,
        backbone: str = "mobilenet_v3_large",
        num_classes: int = 6,
        pretrained: bool = True,
        optimizer: str = "adamw",
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        momentum: float = 0.9,
        programador: str = "coseno",
        paciencia_lr: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = crear_red(backbone, num_classes, pretrained)
        self.criterion = nn.CrossEntropyLoss()

        kw = dict(task="multiclass", num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(average="macro", **kw)
        self.val_f1 = torchmetrics.F1Score(average="macro", **kw)
        self.val_acc = torchmetrics.Accuracy(**kw)
        self.test_f1 = torchmetrics.F1Score(average="macro", **kw)
        self.test_acc = torchmetrics.Accuracy(**kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _paso(self, batch):
        x, y = batch
        logits = self(x)
        return self.criterion(logits, y), logits.argmax(dim=1), y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._paso(batch)
        self.train_f1(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._paso(batch)
        self.val_f1(preds, y)
        self.val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._paso(batch)
        self.test_f1(preds, y)
        self.test_acc(preds, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        h = self.hparams
        if h.optimizer == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=h.lr, weight_decay=h.weight_decay)
        elif h.optimizer == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=h.lr, weight_decay=h.weight_decay)
        elif h.optimizer == "radam":
            opt = torch.optim.RAdam(self.parameters(), lr=h.lr, weight_decay=h.weight_decay)
        elif h.optimizer == "sgd":
            opt = torch.optim.SGD(self.parameters(), lr=h.lr, momentum=h.momentum,
                                  weight_decay=h.weight_decay)
        else:
            raise ValueError("optimizer debe ser 'adamw', 'adam', 'radam' o 'sgd'")

        if h.programador == "coseno":
            programador = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": programador, "interval": "epoch"}}
        if h.programador == "meseta":
            programador = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=h.paciencia_lr)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": programador, "interval": "epoch",
                                                       "monitor": "val/loss"}}
        raise ValueError("programador debe ser 'coseno' o 'meseta'")
