import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
import math
from omegaconf import DictConfig
from typing import Dict, Tuple
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

LOG2 = math.log(2)


class BaseMemory(nn.Module):
    """Base clustering module."""

    def __init__(self, config: DictConfig, span_emb_size: int, encoder_hidden_size: int,drop_module: nn.Module):
        super(BaseMemory, self).__init__()
        self.config = config

        self.mem_size = span_emb_size
        self.encoder_hidden_size = encoder_hidden_size
        # print("Encoder hidden size:",self.encoder_hidden_size)
        self.drop_module = drop_module
        
        self.lqueries =  nn.Embedding(self.config.num_lqueries, self.encoder_hidden_size)
        self.new_entity = nn.Embedding(self.config.num_lqueries, self.encoder_hidden_size)
        self.cls = nn.Embedding(1, self.encoder_hidden_size)
        
        # self.relator = nn.MultiheadAttention(self.mem_size, config.relator_heads)
        print("Encoder Hidden Sixe for Transformer:",self.encoder_hidden_size)
        entity_decoder_layer = nn.TransformerDecoderLayer(d_model=self.encoder_hidden_size, nhead=config.entity_decoder_heads,batch_first=True)
        self.entity_decoder = nn.TransformerDecoder(entity_decoder_layer, num_layers=config.entity_decoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.encoder_hidden_size, nhead=config.decoder_heads,batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        
        self.mem_coref_mlp = MLP(
            self.encoder_hidden_size,
            config.mlp_size,
            1,
            drop_module=drop_module,
            num_hidden_layers=config.mlp_depth,
            bias=True,
        )
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing_wt
        )

    @property
    def device(self) -> torch.device:
        return next(self.mem_coref_mlp.parameters()).device

    def initialize_memory(
        self,
        entity_mentions = None,
        memory = None,
        ent_counter: Tensor = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Method to initialize the clusters and related bookkeeping variables."""
        # Check for unintialized memory
        if entity_mentions is None or ent_counter is None:
            entity_mentions = []
            memory  = []
            ent_counter = torch.tensor([0.0]).to(self.device)

        return entity_mentions,memory,ent_counter

    @staticmethod
    def assign_cluster(coref_new_scores: Tensor) -> Tuple[int, str]:
        """Decode the action from argmax of clustering scores"""

        num_ents = coref_new_scores.shape[0] - 1
        pred_max_idx = torch.argmax(coref_new_scores).item()
        if pred_max_idx < num_ents:
            # Coref
            return pred_max_idx, "c"
        else:
            # New cluster
            return num_ents, "o"
