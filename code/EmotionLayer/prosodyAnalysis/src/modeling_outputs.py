from transformers.file_utils import ModelOutput #type: ignore
from dataclasses import dataclass
from typing import Optional, Tuple
import torch #type: ignore

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
