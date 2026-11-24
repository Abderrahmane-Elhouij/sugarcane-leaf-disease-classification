import timm
import torch
from torch import nn

from src.config import CLASSES, DROP_RATE, LAST_STAGE, PRETRAINED_MODELS, SCRATCH_MODEL, STATS_FILE
from src.utils import load_json


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def classification_head(in_features, num_classes):
    return nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(DROP_RATE),
        nn.Linear(in_features, num_classes),
    )


class PretrainedClassifier(nn.Module):
    def __init__(self, name, num_classes, pretrained=True):
        super().__init__()
        self.name = name
        self.backbone = timm.create_model(PRETRAINED_MODELS[name], pretrained=pretrained, num_classes=0)
        config = self.backbone.pretrained_cfg
        self.normalize = Normalize(config["mean"], config["std"])
        self.embedding_dim = self.backbone.head_hidden_size
        self.head = classification_head(self.embedding_dim, num_classes)

    def forward_features(self, x):
        return self.backbone.forward_features(self.normalize(x))

    def forward_head(self, features, pre_logits=False):
        pooled = self.backbone.forward_head(features, pre_logits=True)
        return pooled if pre_logits else self.head(pooled)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    def embed(self, x):
        return self.forward_head(self.forward_features(x), pre_logits=True)

    def new_parameters(self):
        return list(self.head.parameters())


def conv_bn_relu(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ScratchCNN(nn.Module):
    def __init__(self, num_classes, mean, std, widths=(32, 64, 128, 256)):
        super().__init__()
        self.name = SCRATCH_MODEL
        self.normalize = Normalize(mean, std)
        layers = [conv_bn_relu(3, widths[0], stride=2)]
        in_channels = widths[0]
        for width in widths:
            layers += [conv_bn_relu(in_channels, width), conv_bn_relu(width, width), nn.MaxPool2d(2)]
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_dim = in_channels
        self.head = classification_head(in_channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward_features(self, x):
        return self.features(self.normalize(x))

    def forward_head(self, features, pre_logits=False):
        pooled = self.pool(features).flatten(1)
        return pooled if pre_logits else self.head(pooled)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    def embed(self, x):
        return self.forward_head(self.forward_features(x), pre_logits=True)

    def new_parameters(self):
        return list(self.parameters())


class FusionHead(nn.Module):
    def __init__(self, dims, num_classes):
        super().__init__()
        self.dims = list(dims)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for dim in self.dims)
        self.dropout = nn.Dropout(DROP_RATE)
        self.classifier = nn.Linear(sum(self.dims), num_classes)

    def forward(self, embeddings):
        parts = torch.split(embeddings, self.dims, dim=1)
        fused = torch.cat([norm(part) for norm, part in zip(self.norms, parts)], dim=1)
        return self.classifier(self.dropout(fused))

    def new_parameters(self):
        return list(self.parameters())


class FusionClassifier(nn.Module):
    def __init__(self, members, head):
        super().__init__()
        self.members = nn.ModuleDict(members)
        self.head = head
        for parameter in self.members.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.members.eval()
        return self

    @torch.no_grad()
    def embed(self, x):
        return torch.cat([member.embed(x).float() for member in self.members.values()], dim=1)

    def forward(self, x):
        return self.head(self.embed(x))


def build_model(name, pretrained=True):
    if name == SCRATCH_MODEL:
        stats = load_json(STATS_FILE)
        return ScratchCNN(len(CLASSES), stats["mean"], stats["std"])
    return PretrainedClassifier(name, len(CLASSES), pretrained=pretrained)


def apply_strategy(model, strategy):
    if strategy in ("full_finetuning", "from_scratch"):
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model
    for parameter in model.parameters():
        parameter.requires_grad = False
    if strategy == "partial_finetuning":
        for prefix in LAST_STAGE[model.name]:
            for parameter in model.backbone.get_submodule(prefix).parameters():
                parameter.requires_grad = True
    for parameter in model.new_parameters():
        parameter.requires_grad = True
    return model


def set_train_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if not any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
                module.eval()


def parameter_groups(model, lr_new_layers, lr_pretrained):
    new_ids = {id(parameter) for parameter in model.new_parameters()}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    new = [parameter for parameter in trainable if id(parameter) in new_ids]
    pretrained = [parameter for parameter in trainable if id(parameter) not in new_ids]
    groups = [{"params": new, "lr": lr_new_layers}]
    if pretrained:
        groups.append({"params": pretrained, "lr": lr_pretrained})
    return groups
