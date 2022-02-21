# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import Flatten
from torch import nn

from mmf.utils.build import build_classifier_layer

_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}


@registry.register_model("lstm_only")
class LSTMOnly(BaseModel):
    """LSTMOnly is a simple model with only LSTM for language modeling.

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
        return "configs/models/lstm_only/defaults.yaml"

    def build(self):
        assert len(self._datasets) > 0
        num_question_choices = registry.get(
            _TEMPLATES["question_vocab_size"].format(self._datasets[0])
        )

        self.text_embedding = nn.Embedding(
            num_question_choices, self.config.text_embedding.embedding_dim
        )
        self.lstm = nn.LSTM(**self.config.lstm)

        # As we generate output dim dynamically, we need to copy the config
        # to update it
        self.classifier_s1 = build_classifier_layer(self.config.classifier_state)
        self.classifier_o1 = build_classifier_layer(self.config.classifier_subject)
        self.classifier_o2 = build_classifier_layer(self.config.classifier_object)

    def forward(self, sample_list):
        self.lstm.flatten_parameters()

        question = sample_list.text

        # Get (h_n, c_n), last hidden and cell state
        _, hidden = self.lstm(self.text_embedding(question))
        # X x B x H => B x X x H where X = num_layers * num_directions
        hidden = hidden[0].transpose(0, 1)

        # X should be 2 so we can merge in that dimension
        assert hidden.size(1) == 2, _CONSTANTS["hidden_state_warning"]

        text_features = torch.cat([hidden[:, 0, :], hidden[:, 1, :]], dim=-1)

        o1_logits = self.classifier_o1(text_features)
        o2_logits = self.classifier_o2(text_features)
        s1_logits = self.classifier_s1(text_features)
        output = {"subject_scores": o1_logits, "object_scores": o2_logits, "state_scores": s1_logits}

        return output
