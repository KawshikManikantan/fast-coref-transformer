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

### Dev docs processed for easier access of information:
"""
dict_keys(['subtoken_vs_token', 'clusters_vs_tbound', 'clusters_vs_stbound', 'subtoken_vs_sentence',
'token_vs_tokenstr', 'mentions_vs_stbound', 'mentions_vs_tbound', 'token_vs_subtoken', 'sentence_vs_subtoken',
'token_vs_sentence', 'mentions_vs_sentence', 'sentence_vs_token', 'sentence_vs_sentencestr', 'stbound_wrt_mentions',
'mentions_vs_mentionstr', 'clusters_vs_mentions', 'mentions_vs_clusters', 'mentions_vs_mentiontype', 'mentions_vs_mentionctgry',
'mentiontype_wrt_tbound', 'mentionctgry_wrt_tbound'])
"""
litbank_jsonlines_path = 'litbank_coref_docs_proc.pkl'
with open(litbank_jsonlines_path, "rb") as pickle_file:
    litbank_coref_docs_proc = pickle.load(pickle_file)
# print(litbank_coref_docs_proc['940_the_last_of_the_mohicans_a_narrative_of_1757_brat_0'].keys())


### Process the whole new powerful jsonl file
sys_output = {}
litbank_dev_eval_file = "../models/joint_best/litbank/dev_gold(eval)_tf.log.jsonl"
keys_tracked = ['pred_mentions', 'pred_mentions_emb', 'mention_scores', 'pred_actions', 'predicted_clusters', 'coref_scores','gt_actions']
with jsonlines.open(litbank_dev_eval_file) as reader:
    for obj in reader:
        copy_dict = {}
        for key in keys_tracked:
            copy_dict[key] = obj[key]
        sys_output[obj['doc_key']] = copy_dict
        
whole_sys_output = {}
litbank_dev_eval_file = "../models/joint_best/litbank/dev_gold(eval)_tf.log.jsonl"
with jsonlines.open(litbank_dev_eval_file) as reader:
    for obj in reader:
        copy_dict = {}
        for key in obj:
            if key != 'doc_key':
                copy_dict[key] = obj[key]
        whole_sys_output[obj['doc_key']] = copy_dict
        
best_model_path = "../models/joint_best/model.pth"
checkpoint = torch.load(best_model_path, map_location="cpu")
config = checkpoint["config"]["model"]
train_config = checkpoint["config"]["trainer"]

train_config_encoder = train_config
config_encoder = config
config_encoder.doc_encoder.transformer.model_str = "allenai/longformer-large-4096"
config_encoder.doc_encoder.finetune = False
drop_module = nn.Dropout(p=train_config_encoder.dropout_rate)

mention_proposer = MentionProposalModule(
            config_encoder, train_config_encoder, drop_module=drop_module
        )
mention_proposer.eval()
tokenizer = mention_proposer.doc_encoder.get_tokenizer()
# print(tokenizer)

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
output_dict = mention_proposer(document,False,True)
# print(output_dict["ments"])
torch.save(output_dict["ment_emb_list"],"init_emb.pth")
torch.save(output_dict["ments"],"ments.pth")