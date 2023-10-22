import pickle
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import log2
from collections import Counter
import pandas as pd
from collections import defaultdict
import jsonlines
import os
from os import path
from torch import nn
import scipy
from sklearn.manifold import TSNE
import plotly.express as px
from model.entity_ranking_model import EntityRankingModel
from omegaconf import OmegaConf
from model.mention_proposal import MentionProposalModule

def process_segment(tokenizer,segment):
    return [tokenizer.cls_token_id] + segment + [tokenizer.sep_token_id]

## Load model that is being analysed
def _initialize_best_model(best_model_path,override_encoder,transformer,override_memory,memory):
    checkpoint = torch.load(best_model_path, map_location="cpu")
    config = checkpoint["config"]
    
    # Copying the saved model config to current config is very important to avoid any issues while
    # loading the saved model. To give an example, model might be saved with the speaker tags
    # (training: experiment=ontonotes_speaker)
    # but the evaluation config might lack this detail (eval: experiment=eval_all)
    # However, overriding the encoder is possible -- This method is a bit hacky but allows for overriding the pretrained
    # transformer model from command line.
    
    if override_encoder == True:
        model_config = config.model
        model_config.doc_encoder.transformer = (transformer)

    # Override memory
    # For e.g., can test with a different bounded memory size
    if override_memory == True:
        model_config = config.model
        model_config.memory = memory

    train_info = checkpoint["train_info"]

    if config.model.doc_encoder.finetune:
        # Load the document encoder params if encoder is finetuned
        doc_encoder_dir = path.join(
            path.dirname(best_model_path),
            config.paths.doc_encoder_dirname,
        )
        if path.exists(doc_encoder_dir):
            config.model.doc_encoder.transformer.model_str = doc_encoder_dir

    model = EntityRankingModel(config.model, config.trainer)
    # Document encoder parameters will be loaded via the huggingface initialization
    model.load_state_dict(checkpoint["model"], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    return model
        
whole_sys_output = {}
litbank_dev_eval_file = "../models/joint_best/litbank/dev_gold(eval)_tf.log.jsonl"
with jsonlines.open(litbank_dev_eval_file) as reader:
    for obj in reader:
        copy_dict = {}
        for key in obj:
            if key != 'doc_key':
                copy_dict[key] = obj[key]
        whole_sys_output[obj['doc_key']] = copy_dict
        
# Load the config from a YAML file
transformer_config = OmegaConf.load("conf/model/doc_encoder/transformer/longformer_joint.yaml")
best_model_path = "../models/joint_best/model.pth"
transformer_config = OmegaConf.load("conf/model/doc_encoder/transformer/longformer_joint.yaml")
memory_config = None
model = _initialize_best_model(best_model_path,True,transformer_config,False,memory_config)
mention_proposer = model.mention_proposer
tokenizer = model.mention_proposer.doc_encoder.get_tokenizer()

model.eval()

## Attention visualisation:
document = {}
doc_name = "6053_evelina_or_the_history_of_a_young_ladys_entrance_into_the_world_brat_0"
keys_from_sys_output = ['sentences', 'sent_len_list', 'clusters', 'subtoken_map', 'orig_tokens', 'dataset_name']
for key in keys_from_sys_output:
    document[key] = whole_sys_output[doc_name][key]

tensorized_sent= [
                torch.unsqueeze(
                    torch.tensor(process_segment(tokenizer,sent), device=mention_proposer.device), dim=0
                )
                for sent in document["sentences"]
            ]
document["tensorized_sent"] = tensorized_sent[0]
with torch.no_grad():
    encoded_doc = model.mention_proposer.doc_encoder(document)
    # print(encoded_doc.shape)
    word_attn = torch.squeeze(model.mention_proposer.mention_attn(encoded_doc), dim=1)  # [T]
    # print(word_attn.shape)