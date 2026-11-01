from nnm.models.pretrained import PretrainedModel
from torch import nn


class TinyModel(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.in_features, config.out_features)

    def forward(self, x):
        return self.linear(x)


class EncoderModel(TinyModel):
    module_name = "encoder"


class DecoderModel(TinyModel):
    module_name = "decoder"
