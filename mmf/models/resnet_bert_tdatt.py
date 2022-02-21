# Copyright (c) Facebook, Inc. and its affiliates.

from copy import deepcopy

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import Flatten, ModalCombineLayer, ReLUWithWeightNormFC
from torch import nn

from mmf.modules.embeddings import (
    ImageFeatureEmbedding,
    MultiHeadImageFeatureEmbedding,
    TextEmbedding,
)

from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)

@registry.register_model("resnet_bert_tdatt")
class ResnetBERTTDAtt(BaseModel):
    """ResnetBERTTDAtt is a model for vision and language tasks with simple image and language encoders paired
    with Top-down attention fusion strategy.
    Passes image through a Resnet, and text through an Bert. Image and text features will
    be feed into computation for the attened image feature and fuses them using
    nonlinear element multiplication. Then, it finally passes the fused representation from a MLP to
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
        return "configs/models/resnet_bert_tdatt/defaults.yaml"

    def build(self):
        assert len(self._datasets) > 0
        self.language_module = build_text_encoder(self.config.text_encoder)
        self.image_language_align = ReLUWithWeightNormFC(self.config.text_feature_dim,
                                                         self.config.image_feature_dim)

        setattr(self, "text_embeddings_out_dim", self.config.image_feature_dim)

        self.vision_module = build_image_encoder(self.config.image_encoder)

        self._init_feature_embeddings("image")
        self._init_combine_layer("image", "text")

        # As we generate output dim dynamically, we need to copy the config
        # to update it
        self.classifier_s1 = build_classifier_layer(self.config.classifier_state)
        self.classifier_o1 = build_classifier_layer(self.config.classifier_subject)
        self.classifier_o2 = build_classifier_layer(self.config.classifier_object)

    def _init_text_embeddings(self, attr="text"):
        if "embeddings" not in attr:
            attr += "_embeddings"

        text_embeddings = []
        text_embeddings_list_config = self.config[attr]

        embeddings_out_dim = 0

        for text_embedding in text_embeddings_list_config:
            embedding_type = text_embedding.type
            embedding_kwargs = copy.deepcopy(text_embedding.params)

            self._update_text_embedding_args(embedding_kwargs)

            embedding = TextEmbedding(embedding_type, **embedding_kwargs)

            text_embeddings.append(embedding)
            embeddings_out_dim += embedding.text_out_dim

        setattr(self, attr + "_out_dim", embeddings_out_dim)
        setattr(self, attr, nn.ModuleList(text_embeddings))

    def _init_feature_embeddings(self, attr):
        feature_dim = self.config[attr + "_feature_dim"]
        setattr(self, attr + "_feature_dim", feature_dim)

        feature_embeddings_list = []

        self.feature_embeddings_out_dim = 0

        feature_embeddings = []
        feature_attn_model_list = self.config[attr + "_feature_embeddings"]

        for feature_attn_model_params in feature_attn_model_list:
            feature_embedding = ImageFeatureEmbedding(
                getattr(self, attr + "_feature_dim"),
                self.text_embeddings_out_dim,
                **feature_attn_model_params,
            )
            feature_embeddings.append(feature_embedding)
            self.feature_embeddings_out_dim += feature_embedding.out_dim

        feature_embeddings = nn.ModuleList(feature_embeddings)
        feature_embeddings_list.append(feature_embeddings)

        self.feature_embeddings_out_dim *= getattr(self, attr + "_feature_dim")

        setattr(
            self, attr + "_feature_embeddings_out_dim", self.feature_embeddings_out_dim
        )
        del self.feature_embeddings_out_dim
        setattr(
            self,
            attr + "_feature_embeddings_list",
            nn.ModuleList(feature_embeddings_list),
        )

    def _init_combine_layer(self, attr1, attr2):
        config_attr = attr1 + "_" + attr2 + "_modal_combine"

        multi_modal_combine_layer = ModalCombineLayer(
            self.config[config_attr].type,
            getattr(self, self._get_embeddings_attr(attr1)),
            getattr(self, self._get_embeddings_attr(attr2)),
            **self.config[config_attr].params,
        )

        setattr(
            self,
            attr1 + "_" + attr2 + "_multi_modal_combine_layer",
            multi_modal_combine_layer,
        )

    def _get_embeddings_attr(self, attr):
        embedding_attr1 = attr
        if hasattr(self, attr + "_embeddings_out_dim"):
            embedding_attr1 = attr + "_embeddings_out_dim"
        else:
            embedding_attr1 = attr + "_feature_embeddings_out_dim"

        return embedding_attr1

    def process_feature_embedding(
        self, attr, image_feature, text_embedding, extra=None
    ):
        if extra is None:
            extra = []
        feature_embeddings = []
        feature_attentions = []

        # Get all of the feature embeddings
        list_attr = attr + "_feature_embeddings_list"
        feature_embedding_models = getattr(self, list_attr)

        # Forward through these embeddings one by one
        for feature_embedding_model in feature_embedding_models:
            for feature_embedding_model in feature_embedding_model:
                inp = (image_feature, text_embedding, None, extra)

                embedding, attention = feature_embedding_model(*inp)
                feature_embeddings.append(embedding)
                feature_attentions.append(attention.squeeze(-1))

        # Concatenate all features embeddings and return along with attention
        feature_embedding_total = torch.cat(feature_embeddings, dim=1)
        return feature_embedding_total, feature_attentions

    def combine_embeddings(self, *args):
        feature_names = args[0]
        feature_embeddings = args[1]

        layer = "_".join(feature_names) + "_multi_modal_combine_layer"
        return getattr(self, layer)(*feature_embeddings)

    def forward(self, sample_list):
        text = sample_list["input_ids"]
        text_features = self.language_module(text)[1]
        text_features = self.image_language_align(text_features)

        image = sample_list.image
        image_features = self.vision_module(image)
        attened_image_features, _ = self.process_feature_embedding(
            "image", image_features, text_features
        )

        joint_embedding = self.combine_embeddings(
            ["image", "text"], [attened_image_features, text_features]
        )

        o1_logits = self.classifier_o1(joint_embedding)
        o2_logits = self.classifier_o2(joint_embedding)
        s1_logits = self.classifier_s1(joint_embedding)
        output = {"subject_scores": o1_logits, "object_scores": o2_logits, "state_scores": s1_logits}

        return output
