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


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=6, hidden_size=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(2)])

    def forward(self, input_ids, cache=None):
        if cache is not None:
            keys = input_ids.float().unsqueeze(-1)
            for layer in cache:
                layer.update(keys, keys)
        return self.head(self.embed(input_ids))
