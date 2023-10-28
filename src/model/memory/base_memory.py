import torch
import torch.nn as nn
from pytorch_utils.modules import MLP
import math
from omegaconf import DictConfig
from typing import Dict, Tuple
from torch import Tensor

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

    @property
    def device(self) -> torch.device:
        return next(self.mem_coref_mlp.parameters()).device

    def initialize_memory(
        self,
        entity_mentions = None,
        memory = None,
        ent_counter: Tensor = None,
        last_mention_start: Tensor = None,
        first_mention_start: Tensor = None,
        **kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Method to initialize the clusters and related bookkeeping variables."""
        # Check for unintialized memory
        if entity_mentions is None or ent_counter is None or last_mention_start is None:
            entity_mentions = []
            memory  = []
            ent_counter = torch.tensor([0.0]).to(self.device)
            last_mention_start = torch.zeros(1).long().to(self.device)
            first_mention_start = torch.zeros(1).long().to(self.device)

        return entity_mentions,memory,ent_counter, last_mention_start,first_mention_start

    def get_coref_new_scores(
        self,
        ment_tok_emb_tensor,
        memory,
    ) -> Tensor:
        """Calculate the coreference score with existing clusters.

        For creating a new cluster we use a dummy score of 0.
        This is a free variable and this idea is borrowed from Lee et al 2017

        Args:
                        ment_tok_emb_tensor(T_{mj}x d'): Mention representation
                        mem_tok_emb_tensor(M x d'): Cluster representations
                        ent_counter (M): Mention counter of clusters.
                        

        Returns:
                        coref_new_score (M + 1):
                                        Coref scores concatenated with the score of forming a new cluster.
        """
        
        ## Memory
        options = memory + [self.new_entity.weight]
        options = torch.stack(options)
        
        ## Query
        query = torch.cat([self.cls.weight, ment_tok_emb_tensor],dim = 0)
        query = query.unsqueeze(dim=0).repeat(options.shape[0],1,1)
        
        # print("Options shape:",options.shape)
        # print("Query Shape:",query.shape)
        try:
            outputs = self.decoder(tgt=query,memory=options)
            # print("Outputs shape:",outputs.shape)
            coref_score = self.mem_coref_mlp(outputs[:,0,:]).squeeze(dim=1)
            # print("Coref score shape:",coref_score.shape)
            # modified_mention = torch.mean(outputs[:,1:,:], dim=0).unsqueeze(dim=0)
        except:
            print("Options shape:",options.shape)
            print("Query Shape:",query.shape)
            raise
        # print("Modified mention shape:",modified_mention.shape)
        # print(modified_mention.shape)
        return coref_score

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
