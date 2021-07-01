import logging
import os

import torch
from hydra.utils import get_original_cwd

import torchvision.models as models

from metro.modeling._mano import MANO
from metro.modeling._mano import Mesh as ManoMesh
from metro.modeling._smpl import SMPL
from metro.modeling._smpl import Mesh as SmplMesh
from metro.modeling.bert import METRO, BertConfig
# from metro.modeling.bert import METRO_Body_Network as METRO
from metro.modeling.bert_v2 import METROBodyNetwork
from metro.modeling.hrnet.config import config as hrnet_config
from metro.modeling.hrnet.config import update_config as hrnet_update_config
from metro.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net

LOG = logging.getLogger(__name__)


def build_model(cfg):
    body_model = cfg.model.BODY_MODEL
    if body_model == "SMPL":
        model = build_smpl_model(cfg)
    elif body_model == "MANO":
        model = build_mano_model(cfg)
    return model


def build_transformer_encoder(cfg):
    # (dennis.park) Read configs
    input_feat_dim = cfg.INPUT_FEAT_DIM
    hidden_feat_dim = cfg.HIDDEN_FEAT_DIM
    dropout_prob = cfg.DROPOUT_PROB
    config_path = cfg.BERT_CONFIG_PATH

    num_hidden_layers = cfg.NUM_HIDDEN_LAYERS
    num_attention_heads = cfg.NUM_ATTENTION_HEADS

    output_feat_dim = input_feat_dim[1:] + [3]

    # (dennis.park) Stack encoder blocks.
    trans_encoder = []
    # init three transformer-encoder blocks in a loop
    for i in range(len(output_feat_dim)):
        config_class, model_class = BertConfig, METRO
        config = config_class.from_pretrained(os.path.join(get_original_cwd(), config_path))

        config.output_attentions = False
        config.hidden_dropout_prob = dropout_prob
        config.img_feature_dim = input_feat_dim[i]
        config.output_feature_dim = output_feat_dim[i]
        hidden_size = hidden_feat_dim[i]

        # We have recently tried to use an updated intermediate size, which is 4*hidden-size.
        # But we didn't find significant performance changes on Human3.6M (~36.7 PA-MPJPE)
        # intermediate_size = hidden_size * 4

        config.num_hidden_layers = num_hidden_layers
        config.num_attention_heads = num_attention_heads
        config.hidden_size = hidden_size
        # config.intermediate_size = intermediate_size

        # init a transformer encoder and append it to a list
        assert config.hidden_size % config.num_attention_heads == 0
        model = model_class(config=config)
        LOG.info("Init model from scratch.")
        trans_encoder.append(model)

    return torch.nn.Sequential(*trans_encoder)


def build_backbone(cfg):
    if cfg.NAME in ('hrnet', 'hrnet-w64'):
        # HRnet backbone.
        hrnet_update_config(hrnet_config, os.path.join(get_original_cwd(), cfg.HRNET_CONFIG))
        backbone = get_cls_net(hrnet_config, pretrained=os.path.join(get_original_cwd(), cfg.HRNET_CKPT))
    else:
        # torchvision model pre-trained on imagenet.
        backbone = models.__dict__[cfg.NAME](pretrained=True)
    return backbone


def build_metro_network(cfg):
    trans_encoder = build_transformer_encoder(cfg.model.TRANSFORMER_ENCODER)
    backbone = build_backbone(cfg.model.backbone)
    # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
    metro_network = METROBodyNetwork(cfg, backbone, trans_encoder)
    return metro_network


def build_smpl_model(cfg):
    metro_network = build_metro_network(cfg)
    smpl = SMPL().to('cuda')
    mesh_sampler = SmplMesh()
    return metro_network, smpl, mesh_sampler


def build_mano_model(cfg):
    metro_network = build_metro_network(cfg)
    mano_model = MANO().to('cuda')
    mano_model.layer = mano_model.layer.cuda()
    mesh_sampler = ManoMesh()
    raise NotImplementedError("Not fully implemented yet.")
