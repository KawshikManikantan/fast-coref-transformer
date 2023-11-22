import torch
from model.memory import BaseMemory
from pytorch_utils.modules import MLP
import torch.nn as nn

from omegaconf import DictConfig
from typing import Dict, Tuple, List
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


class EntityMemory(BaseMemory):
    """Module for clustering proposed mention spans using Entity-Ranking paradigm."""

    def __init__(
        self, config: DictConfig, span_emb_size: int,encoder_hidden_size: int, drop_module: nn.Module
    ) -> None:
        super(EntityMemory, self).__init__(config, span_emb_size,encoder_hidden_size, drop_module)
        self.mem_type: DictConfig = config.mem_type
    
    def get_new_entity(self,entity_mentions):
        # print("Entity mentytions shape:", entity_mentions.shape)
        outputs = self.entity_decoder(tgt=self.lqueries.weight,memory=entity_mentions)
        
        # print("Decoder output shape:", outputs.shape)
        return outputs

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
        entity_options = memory + [self.new_entity.weight]
        entity_options = torch.stack(entity_options)
        query_seed = torch.cat([self.cls.weight, ment_tok_emb_tensor],dim = 0)
        
        entity_dataset = TensorDataset(entity_options)
        dataloader = DataLoader(entity_dataset, batch_size=self.config.max_batch_size)
        coref_score_final = torch.zeros(len(entity_options)).to(self.device)
        num_processed = 0  

        ## Query
        for entity_batch in dataloader:
            options = entity_batch[0]
            query = query_seed.unsqueeze(dim=0).repeat(options.shape[0],1,1)
            try:
                outputs = self.decoder(tgt=query,memory=options)
                score_tensor = outputs[:,0,:].clone()
                # print(len(entity_options),num_processed,num_processed+options.shape[0])
                coref_score_final[num_processed:num_processed+options.shape[0]] = self.mem_coref_mlp(score_tensor).squeeze(dim=1)
            except:
                print("Options shape:",options.shape)
                print("Query Shape:",query.shape)
                breakpoint()
                raise
            
            num_processed += options.shape[0]
        
        
        return coref_score_final 
        
    def forward_training(
        self,
        ment_boundaries: Tensor,
        ment_tok_emb_list: List[Tensor],
        gt_actions: List[Tuple[int, str]],
        metadata: Dict,
    ) -> List[Tensor]:
        """
        Forward pass during coreference model training where we use teacher-forcing.

        Args:
                ment_boundaries: Mention boundaries of proposed mentions
                mention_emb_list: Embedding list of proposed mentions
                gt_actions: Ground truth clustering actions
                metadata: Metadata such as document genre

        Returns:
                coref_new_list: Logit scores for ground truth actions.
        """
        # Initialize memory
        first_overwrite = True
        entity_mentions,memory,ent_counter = self.initialize_memory()
        coref_loss = None
        num_ents = 0
        for ment_idx, (ment_tok_emb_tensor, (gt_cell_idx, gt_action_str)) in enumerate(
            zip(ment_tok_emb_list, gt_actions)
        ):
            if ment_idx > 350:
                break
            
            ment_start, ment_end = ment_boundaries[ment_idx]
            coref_new_scores = None
            if not first_overwrite:
                coref_new_scores = self.get_coref_new_scores(
                    ment_tok_emb_tensor, memory
                )
            else:
                first_overwrite = False

            # Teacher forcing
            # coref_new_scores_1 = torch.randn(1024, device=self.device)
            action_str, cell_idx = gt_action_str, gt_cell_idx
            if action_str == "c":
                entity_mentions[cell_idx] = torch.cat([entity_mentions[cell_idx], ment_tok_emb_tensor],dim=0)
                memory[cell_idx] = self.get_new_entity(entity_mentions[cell_idx])
                ent_counter[cell_idx] = ent_counter[cell_idx] + 1

            elif action_str == "o":
                entity_mentions.append(ment_tok_emb_tensor)
                memory.append(self.get_new_entity(entity_mentions[-1]))
                ent_counter = torch.cat(
                    [ent_counter, torch.tensor([1.0], device=self.device)], dim=0
                )
                num_ents += 1

            if coref_new_scores is not None:
                target = torch.tensor([gt_cell_idx], device=self.device)
                if coref_loss is None:
                    coref_loss = self.loss_fn(torch.unsqueeze(coref_new_scores, dim=0), target)
                    # coref_loss.backward(retain_graph=True)
                    # torch.cuda.empty_cache()
                else:
                    coref_loss += self.loss_fn(torch.unsqueeze(coref_new_scores, dim=0), target)
                    # torch.cuda.empty_cache()
                    # coref_loss.backward(retain_graph=True)
        # breakpoint()  
        # del entity_mentions
        # del memory
        # import gc
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(obj.name, type(obj), obj.size())
        #     except:
        #         pass
        print("Peak:",torch.cuda.memory_allocated()/1048576)
        # breakpoint()
        # print(coref_loss)
        return coref_loss

    def forward(
        self,
        ment_boundaries: Tensor,
        mention_tok_emb_list: List[Tensor],
        gt_actions:  List[Tuple[int, str]],
        teacher_force: False,
        memory_init: Dict = None,
    ):
        """Forward pass for clustering entity mentions during inference/evaluation.

        Args:
         ment_boundaries: Start and end token indices for the proposed mentions.
         mention_emb_list: Embedding list of proposed mentions
         metadata: Metadata features such as document genre embedding
         memory_init: Initializer for memory. For streaming coreference, we can pass the previous
                  memory state via this dictionary

        Returns:
                pred_actions: List of predicted clustering actions.
                mem_state: Current memory state.
        """

        assert len(mention_tok_emb_list) == len(gt_actions)
        # Initialize memory
        if memory_init is not None:
            entity_mentions, memory, ent_counter = self.initialize_memory(
                **memory_init
            )
        else:
            entity_mentions, memory, ent_counter = self.initialize_memory()

        pred_actions = []  # argmax actions
        coref_scores_list = []
        
        # Boolean to track if we have started tracking any entities
        # This value can be false if we are processing subsequent chunks of a long document
        first_overwrite: bool = True if torch.sum(ent_counter) == 0 else False
    
        for ment_idx, ment_tok_emb_tensor in enumerate(mention_tok_emb_list):
            ment_start, ment_end = ment_boundaries[ment_idx]

            if first_overwrite:
                pred_cell_idx, pred_action_str = 0, "o"
            else:
                coref_new_scores = self.get_coref_new_scores(
                    ment_tok_emb_tensor, memory 
                )
                coref_scores_list.append(coref_new_scores.detach().cpu())
                pred_cell_idx, pred_action_str = self.assign_cluster(coref_new_scores)

            if teacher_force:
                next_cell_idx,next_action_str = gt_actions[ment_idx]
                pred_actions.append(gt_actions[ment_idx])
            else:
                next_cell_idx,next_action_str = pred_cell_idx, pred_action_str
                pred_actions.append((pred_cell_idx, pred_action_str))
                
            if first_overwrite:
                first_overwrite = False
                # We start with a single empty memory cell
                entity_mentions.append(ment_tok_emb_tensor)
                memory.append(self.get_new_entity(ment_tok_emb_tensor))
                ent_counter = torch.tensor([1.0], device=self.device)
            else:
                if next_action_str == "c":
                    # Perform coreference update on the cluster referenced by pred_cell_idx
                    entity_mentions[next_cell_idx] = torch.cat([entity_mentions[next_cell_idx],ment_tok_emb_tensor],dim=0)
                    memory[next_cell_idx] = self.get_new_entity(entity_mentions[next_cell_idx])
                    ent_counter[next_cell_idx] = ent_counter[next_cell_idx] + 1

                elif next_action_str == "o":
                    # Append the new entity to the entity cluster array
                    entity_mentions.append(ment_tok_emb_tensor)
                    memory.append(self.get_new_entity(ment_tok_emb_tensor))
                    ent_counter = torch.cat(
                        [ent_counter, torch.tensor([1.0], device=self.device)], dim=0
                    )

        mem_state = {
            "entity_mentions": entity_mentions,
            "memory": memory,
            "ent_counter": ent_counter,
        }
        return pred_actions, mem_state, coref_scores_list
