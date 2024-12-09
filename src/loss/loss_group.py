
from torch import nn



class LossGroup(nn.Module):
    name: str


    def __init__(
        self,
        name: str,

    ) -> None:
        super().__init__()
        self.name = name

