# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import requests
import torch
from mmf.common.report import Report
from mmf.common.sample import Sample, SampleList
from mmf.utils.build import build_encoder, build_model, build_processors
from mmf.utils.checkpoint import load_pretrained_model
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf, DictConfig
from PIL import Image

from mmf.utils.build import (
    build_classifier_layer,
    build_image_encoder,
    build_text_encoder,
)


class Inference_R_B_C:
    def __init__(self, checkpoint_path: str = None, config: DictConfig = None):
        self.checkpoint = checkpoint_path
        self.config = config
        assert self.checkpoint is not None
        self.processor, self.model = self._build_model() # self.feature_extractor,

    def _build_model(self):
        self.model_items = load_pretrained_model(self.checkpoint)
        # self.config = OmegaConf.create(self.model_items["full_config"])
        dataset_name = list(self.config.dataset_config.keys())[0]
        processor = build_processors(
            self.config.dataset_config[dataset_name].processors
        )
        # feature_extractor = build_encoder(
        #     self.model_items["config"].image_feature_encodings
        # )
        ckpt = self.model_items["checkpoint"]
        model = build_model(self.config.model_config.resnet_bert_concat)
        model.load_state_dict(ckpt)

        return processor, model  # feature_extractor

    def forward(self, image_path: str, text: dict, image_format: str = "path"):
        text_output = self.processor["text_processor"]({"text": text})

        if image_format == "path":
            # img = np.array(Image.open(image_path))
            img = Image.open(image_path)
        elif image_format == "url":
            img = np.array(Image.open(requests.get(image_path, stream=True).raw))

        image_preprocessed = self.processor["image_processor"](img)
        image_preprocessed = torch.as_tensor(image_preprocessed)

        sample = Sample()
        sample.text = text_output["text"]
        if "input_ids" in text_output:
            sample.update(text_output)
        sample.image = image_preprocessed
        sample_list = SampleList([sample])
        sample_list = sample_list.to(get_current_device())
        self.model = self.model.to(get_current_device())
        self.model.eval()
        model_output = self.model(sample_list)

        state_outputs = model_output["state_scores"]
        subject_outputs = model_output["subject_scores"]
        object_outputs = model_output["object_scores"]
        if state_outputs.dim() == 2:
            state_outputs = state_outputs.topk(1, 1, True, True)[1].t().squeeze()
        if subject_outputs.dim() == 2:
            subject_outputs = subject_outputs.topk(1, 1, True, True)[1].t().squeeze()
        if object_outputs.dim() == 2:
            object_outputs = object_outputs.topk(1, 1, True, True)[1].t().squeeze()

        action = self.processor["answer_processor"].state_idx2word(state_outputs.item())
        subject = self.processor["answer_processor"].obj_idx2word(subject_outputs.item())
        object = self.processor["answer_processor"].obj_idx2word(object_outputs.item())

        return action, subject, object
