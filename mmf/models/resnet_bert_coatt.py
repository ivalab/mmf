# Copyright (c) Facebook, Inc. and its affiliates.

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import TwoBranchCombineLayer, ClassifierLayer, Flatten, ModalCombineLayer, ReLUWithWeightNormFC
from torch import nn

from mmf.modules.embeddings import (
    ImageFeatureEmbedding,
    MultiHeadImageFeatureEmbedding,
    PreExtractedEmbedding,
    TextEmbedding,
    MCANEmbedding,
)

from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)

@registry.register_model("resnet_bert_coatt")
class ResnetBERTCOAtt(BaseModel):
    """ResnetBERTCOAtt is a model for vision and language tasks with simple image and language encoders paired
    with Co-Attention fusion strategy.
    Passes image through a Resnet, and text through an Bert. We employ co-attention as the fusion module here.
    Then, it finally passes the fused representation from a MLP to
    generate scores for each of the possible answers.

    Args:
        config (DictConfig): Configuration node containing all of the necessary
                             config required to initialize ResnetLSTMConcat.

    Inputs: sample_list (SampleList)
        - **sample_list** should contain image attribute for image, text for
          question split into word indices, targets for answer scores
    """

    def __init__(self, config):
        super().__init__(config)
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/resnet_bert_coatt/defaults.yaml"

    def build(self):
        assert len(self._datasets) > 0

        self.image_feature_dim = self.config["image_feature_dim"]

        self.language_module = build_text_encoder(self.config.text_encoder)
        self.image_language_align = ReLUWithWeightNormFC(self.config.text_feature_dim,
                                                         self.config.image_feature_dim)

        setattr(self, "text_embeddings_out_dim", self.config.image_feature_dim)

        self.vision_module = build_image_encoder(self.config.image_encoder)

        # Text encoder
        self._init_text_embeddings("text")

        # Text-image decoder
        self._init_feature_embeddings("image")

        # Fusion
        self._init_combine_layer("image", "text")

        # As we generate output dim dynamically, we need to copy the config
        # to update it
        self.classifier_s1 = build_classifier_layer(self.config.classifier_state)
        self.classifier_o1 = build_classifier_layer(self.config.classifier_subject)
        self.classifier_o2 = build_classifier_layer(self.config.classifier_object)

    # Directly from movie_mcan
    def _init_text_embeddings(self, attr="text"):
        if "embeddings" not in attr:
            attr += "_embeddings"

        module_config = self.config[attr]
        embedding_type = module_config.type
        embedding_kwargs = copy.deepcopy(module_config.params)
        self._update_text_embedding_args(embedding_kwargs)
        embedding = TextEmbedding(embedding_type, **embedding_kwargs)
        embeddings_out_dim = embedding.text_out_dim

        setattr(self, attr + "_out_dim", embeddings_out_dim)
        setattr(self, attr, embedding)

    def _update_text_embedding_args(self, args):
        # Add model_data_dir to kwargs
        args.model_data_dir = self.config.model_data_dir

    # We use MCANEmbedding
    def _init_feature_embeddings(self, attr: str):
        embedding_kwargs = self.config[attr + "_feature_embeddings"]["params"]
        setattr(
            self, attr + "_feature_embeddings_out_dim", embedding_kwargs["hidden_dim"]
        )
        assert (
            getattr(self, attr + "_feature_embeddings_out_dim")
            == self.text_embeddings_out_dim
        ), "dim1: {}, dim2: {}".format(
            getattr(self, attr + "_feature_embeddings_out_dim"),
            self.text_embeddings_out_dim,
        )

        feature_embedding = MCANEmbedding(
            getattr(self, attr + "_feature_dim"), **embedding_kwargs
        )
        setattr(self, attr + "_feature_embeddings_list", feature_embedding)

    def _get_embeddings_attr(self, attr):
        embedding_attr1 = attr
        if hasattr(self, attr + "_embeddings_out_dim"):
            embedding_attr1 = attr + "_embeddings_out_dim"
        else:
            embedding_attr1 = attr + "_feature_embeddings_out_dim"

        return embedding_attr1

    def _init_combine_layer(self, attr1: str, attr2: str):
        multi_modal_combine_layer = TwoBranchCombineLayer(
            getattr(self, self._get_embeddings_attr(attr1)),
            getattr(self, self._get_embeddings_attr(attr2)),
        )

        setattr(
            self,
            attr1 + "_" + attr2 + "_multi_modal_combine_layer",
            multi_modal_combine_layer,
        )

    def process_text_embedding(
        self, text_features, text_mask, embedding_attr: str = "text_embeddings"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get embedding models
        text_embedding_model = getattr(self, embedding_attr)

        text_embedding_total, text_embedding_vec = text_embedding_model(
            text_features, text_mask
        )

        return text_embedding_total, text_embedding_vec

    def process_feature_embedding(
        self,
        attr: str,
        image_features,
        text_mask,
        text_embedding_total: torch.Tensor,
    ):
        feature_embedding = getattr(self, attr + "_feature_embeddings_list")
        feature_sga = feature_embedding(
            image_features,
            text_embedding_total,
            None,
            text_mask,
        )

        return feature_sga

    def combine_embeddings(self, *args):
        feature_names = args[0]
        feature_embeddings = args[1]

        layer = "_".join(feature_names) + "_multi_modal_combine_layer"
        return getattr(self, layer)(*feature_embeddings)

    def forward(self, sample_list):

        text = sample_list["input_ids"]
        text_mask = sample_list["input_mask"]

        text_mask = text_mask.gt(0) # Convert to bool

        text_features = self.language_module(text)[0]

        # b*length*hidden_dim
        text_embedding_total, text_embedding_vec = self.process_text_embedding(
            text_features, text_mask
        )

        image = sample_list.image
        image_features = self.vision_module(image)

        # b*(h*w)*hidden_dim
        feature_sga = self.process_feature_embedding(
            "image", image_features, text_mask, text_embedding_total
        )

        # b*2*(h*w)*(hidden_dim*2)
        joint_embedding = self.combine_embeddings(
            ["image", "text"], [feature_sga, text_embedding_vec[:, 1]]
        )

        o1_logits = self.classifier_o1(joint_embedding)
        o2_logits = self.classifier_o2(joint_embedding)
        s1_logits = self.classifier_s1(joint_embedding)
        output = {"subject_scores": o1_logits, "object_scores": o2_logits, "state_scores": s1_logits}

        return output
