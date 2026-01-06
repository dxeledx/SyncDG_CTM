from syncdg_ctm.losses.cdan import cdan_onehot_loss
from syncdg_ctm.losses.coral import CovarianceEMAMemory, coral_ema_loss, coral_loss
from syncdg_ctm.losses.diversity import tick_diversity_loss
from syncdg_ctm.losses.orth import orth_loss
from syncdg_ctm.losses.proto import PrototypeEMAMemory, proto_alignment_loss
from syncdg_ctm.losses.supcon import supervised_contrastive_loss

__all__ = [
    "CovarianceEMAMemory",
    "PrototypeEMAMemory",
    "cdan_onehot_loss",
    "coral_ema_loss",
    "coral_loss",
    "orth_loss",
    "proto_alignment_loss",
    "supervised_contrastive_loss",
    "tick_diversity_loss",
]
