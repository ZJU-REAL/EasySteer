import torch
from collections import OrderedDict

from ...core.interventions import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
)


class BiasIntervention(
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention
):
    """Add a trainable bias to the hidden state: h + b."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.bias = torch.nn.Parameter(
            torch.zeros(self.embed_dim), requires_grad=True
        )
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)

    def forward(self, base, source=None, subspaces=None):
        output = base + self.bias
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """Return the learned bias in the ReFT checkpoint format."""
        state_dict = OrderedDict()
        state_dict["bias"] = self.bias.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load the saved bias onto the intervention's current device."""
        if "bias" in state_dict:
            self.bias.data = state_dict["bias"].to(self.bias.device)