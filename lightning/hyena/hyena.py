import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from einops import rearrange

from safari.models.sequence.simple_lm import SimpleLMHeadModel


class Hyena(pl.LightningModule):
    def __init__(self, lr, **kwargs):
        super().__init__()
        self.lr = lr
        self.hyena = SimpleLMHeadModel(**kwargs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        output, _ = self.hyena(input_ids=inputs)
        logits = output.logits

        # logits: [b, s, v] -> [v, b*s]
        logits = rearrange(logits, "b s v -> (b s) v")

        # targets: [b, s] -> [b*s]
        targets = rearrange(targets, "b s -> (b s)")

        loss = F.cross_entropy(logits, targets, ignore_index=-100)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
