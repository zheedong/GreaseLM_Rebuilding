import os

import torch
import troch.nn as nn
import torch.nn.functional as F
from transformers import (
    modeling_bert,
    BertConfig,
    modeling_roberta,
    RobertaConfig,
    PretrainedConfig,
    WEIGHTS_NAME,
)
from transformers.configuration_utils import (
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    cached_property,
    hf_bucket_url,
    is_remote_url,
)
import logging

from modeling import modeling_gnn
from utils import layers
from utils import utils

logger = logging.getLogger(__name__)

if os.environ.get('INHERIT_BERT', 0):
    ModelClass = modeling_bert.BertModel
else:
    ModelClass = modeling_roberta.RobertaModel

print(f'ModelClass: {ModelClass}')


class GreaseLM(nn.Module):
    def __init__(
            self,
            args={},
            model_name="roberta-base",
            k=5,
            n_ntype=4,
            n_etype=38,
            n_concepts=799273,
            concept_dim=200,
            concept_int_dim=1024,
            n_attention_head=2,
            fc_dim=200,
            n_fc_layer=0,
            p_emb=0.2,
            p_gnn=0.2,
            p_cf=0.2,
            pretrained_concept_emb=None,
            freeze_ent_emb=True,
            init_range=0.02,
            ie_dim=200,
            info_exchange=True,
            ie_layer_num=1,
            sep_ie_layers=False,
            layer_id=-1):
        super().__init__()
        self.lmgnn = LMGNN(
            args,
            model_name,
            k,
            n_ntype,
            n_etype,
            n_concepts,
            concept_dim,
            concept_int_dim,
            n_attention_head,
            fc_dim,
            n_fc_layer,
            p_emb,
            p_gnn,
            p_cf,
            pretrained_concept_emb,
            freeze_ent_emb,
            init_range,
            ie_dim,
            info_exchange,
            ie_layer_num,
            sep_ie_layers,
            layer_id)

        def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
            raise NotImplementedError

        def forward(self, *inputs, cache_output=False, detail=False):
            raise NotImplementedError

        def get_fake_inputs(self, device="cuda:0"):
            raise NotImplementedError

        def get_fake_inputs(self, device="cuda:0"):
            raise NotImplementedError

        def check_outputs(self, logits, attn):
            raise NotImplementedError


def test_GreaseLM():
    raise NotImplementedError


class LMGNN(nn.Module):
    def __init__(
            self,
            args={},
            model_name="roberta-base",
            k=5,
            n_ntype=4,
            n_etype=38,
            n_concepts=799273,
            concept_dim=200,
            concept_int_dim=1024,
            n_attention_head=2,
            fc_dim=200,
            n_fc_layer=0,
            p_emb=0.2,
            p_gnn=0.2,
            p_cf=0.2,
            pretrained_concept_emb=None,
            freeze_ent_emb=True,
            init_range=0.02,
            ie_dim=200,
            info_exchange=True,
            ie_layer_num=1,
            sep_ie_layers=False,
            layer_id=-1):
        super().__init__()
        raise NotImplementedError

    def _init_weights(self, module):
        raise NotImplementedError

    def forward(
            self,
            inputs,
            concept_ids,
            node_type_ids,
            node_scores,
            adj_lengths,
            special_nodes_mask,
            adj,
            emb_data=None,
            cache_output=False):
        raise NotImplementedError

    def get_fake_inputs(self, device="cuda:0"):
        raise NotImplementedError

    def check_outputs(self, logits, pool_attn):
        raise NotImplementedError


class TextKGMessagePassing(ModelClass):
    def __init__(
            self,
            config,
            args={},
            k=5,
            n_ntype=4,
            n_etype=38,
            dropout=0.2,
            concept_dim=200,
            ie_dim=200,
            p_fc=0.2,
            info_exchange=True,
            ie_layer_num=1,
            sep_ie_layers=False):
        raise NotImplementedError

    def _init_weights(self, module):
        raise NotImplementedError

    def forward(self, input_ids, token_type_ids, attention_mask, special_tokens_mask, H, A, node_type, node_score, special_nodes_mask, cache_output=False, position_ids=None, head_mask=None, output_hidden_states=True):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        raise NotImplementedError

    def get_fake_inputs(self, device="cuda:0"):
        raise NotImplementedError

    def check_outputs(self, outputs, gnn_output):
        raise NotImplementedError


class RoBERTaGAT(modeling_bert.BertEncoder):

    def __init__(self, config, k=5, n_ntype=4, n_etype=38, hidden_size=200, dropout=0.2, concept_dim=200, ie_dim=200, p_fc=0.2, info_exchange=True, ie_layer_num=1, sep_ie_layers=False):
        raise NotImplementedError

    def forward(self, hidden_states, attention_mask, special_tokens_mask, head_mask, _X, edge_index, edge_type, _node_type, _node_feature_extra, special_nodes_mask, output_attentions=False, output_hidden_states=True):
        raise NotImplementedError

    def get_fake_inputs(self, device="cuda:0"):
        raise NotImplementedError

    def check_outputs(self, outputs, _X):
        raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    utils.print_cuda_info()
    free_gpus = utils.select_free_gpus()
    device = torch.device("cuda:{}".format(free_gpus[0]))

    # test_RoBERTaGAT(device)

    # test_TextKGMessagePassing(device)

    # test_LMGNN(device)

    test_GreaseLM(device)

