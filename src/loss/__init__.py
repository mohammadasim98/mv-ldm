from dataclasses import dataclass

from .loss_group import LossGroup

from .loss_discriminator import LossDiscriminator, LossDiscriminatorCfg
from .loss_depth import LossDepth, LossDepthCfg
from .loss_generator import LossGenerator, LossGeneratorCfg
from .loss_isotropy import LossIsotropy, LossIsotropyCfg
from .loss_kl import LossKl, LossKlCfg
from .loss_l1 import LossL1, LossL1Cfg
from .loss_lpips import LossLpips, LossLpipsCfg
from .loss_mse import LossMse, LossMseCfg
from .loss_diffusion import LossDiffusion, LossDiffusionCfg


LOSSES = {
    "depth": LossDepth,
    "isotropy": LossIsotropy,
    "kl": LossKl,
    "l1": LossL1,
    "lpips": LossLpips,
    "mse": LossMse
}


NLLLossCfg = LossDepthCfg | LossIsotropyCfg | LossKlCfg | LossL1Cfg | LossLpipsCfg | LossMseCfg

