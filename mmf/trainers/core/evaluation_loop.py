# Copyright (c) Facebook, Inc. and its affiliates.
import os
import logging
from abc import ABC
from typing import Any, Dict, Tuple, Type
import time

import torch
import tqdm
from caffe2.python.timeout_guard import CompleteInTimeOrDie
from mmf.common.meter import Meter
from mmf.common.report import Report
from mmf.common.sample import to_device
from mmf.utils.distributed import gather_tensor, is_main, is_xla

import cv2

logger = logging.getLogger(__name__)


class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, dataset_type: str, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        use_cpu = self.config.evaluation.get("use_cpu", False)
        visualize = self.config.evaluation.get("visualize", True)
        loaded_batches = 0
        skipped_batches = 0

        vocab_state = {}
        vocab_object = {}
        if visualize:
            cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

            with open('/home/ruinian/.cache/torch/mmf/data/datasets/sgl/defaults/extras/vocabs/state_sgl.txt',
                      'r') as f:
                state = f.readline()
                idx = 0
                while state:
                    vocab_state[idx] = state

                    idx += 1
                    state = f.readline()

            with open('/home/ruinian/.cache/torch/mmf/data/datasets/sgl/defaults/extras/vocabs/object_sgl.txt',
                      'r') as f:
                obj = f.readline()
                idx = 0
                while obj:
                    vocab_object[idx] = obj

                    idx += 1
                    obj = f.readline()

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_main()
            while reporter.next_dataset(flush_report=False):
                dataloader = reporter.get_dataloader()
                combined_report = None

                correct_prediction = 0
                total_sample = 0

                if self._can_use_tqdm(dataloader):
                    dataloader = tqdm.tqdm(dataloader, disable=disable_tqdm)

                total_time_cost = 0.
                for batch in dataloader:
                    # Do not timeout quickly on first batch, as workers might start at
                    # very different times.
                    with CompleteInTimeOrDie(600 if loaded_batches else 3600 * 24):
                        loaded_batches += 1
                        prepared_batch = reporter.prepare_batch(batch)
                        prepared_batch = to_device(prepared_batch, self.device)
                        if not validate_batch_sizes(prepared_batch.get_batch_size()):
                            logger.info("Skip batch due to uneven batch sizes.")
                            skipped_batches += 1
                            continue

                        start = time.time()
                        model_output = self.model(prepared_batch)
                        total_time_cost += time.time() - start

                        report = Report(prepared_batch, model_output)
                        report = report.detach()

                        batch_size = prepared_batch['image_id'].size()[0]
                        total_sample += batch_size
                        for idx in range(batch_size):
                            state_score = model_output['state_scores'][idx, :]
                            subject_score = model_output['subject_scores'][idx, :]
                            object_score = model_output['object_scores'][idx, :]
                            state_target = prepared_batch['state_targets'][idx, 0]
                            subject_target = prepared_batch['subject_targets'][idx, 0]
                            object_target = prepared_batch['object_targets'][idx, 0]

                            func = torch.nn.Softmax(dim=0)
                            state_score = func(state_score)
                            subject_score = func(subject_score)
                            object_score = func(object_score)
                            state_pred = torch.argmax(state_score, dim=0)
                            subject_pred = torch.argmax(subject_score, dim=0)
                            object_pred = torch.argmax(object_score, dim=0)

                            if state_target == state_pred and subject_target == subject_pred and \
                                object_target == object_pred:
                                correct_prediction += 1

                            if visualize:
                                image_id = prepared_batch['image_id'][idx]
                                image = cv2.imread(os.path.join(
                                    '/home/ruinian/.cache/torch/mmf/data/datasets/sgl/defaults/images/val',
                                    '{}.png'.format(image_id)))
                                natural_language = prepared_batch['text'][idx][1:-1]

                                cv2.imshow('image', image)
                                print(' '.join(natural_language[:]))

                                print("Groundtruth: PDDL goal state: {} {} {}".format(
                                    vocab_state[int(state_target.cpu().numpy())],
                                    vocab_object[int(subject_target.cpu().numpy())],
                                    vocab_object[int(object_target.cpu().numpy())]))
                                print("Prediction: PDDL goal state: {} {} {}".format(
                                    vocab_state[int(state_pred.cpu().numpy())],
                                    vocab_object[int(subject_pred.cpu().numpy())],
                                    vocab_object[int(object_pred.cpu().numpy())]))
                                cv2.waitKey(1)

                        meter.update_from_report(report)

                        moved_report = report
                        # Move to CPU for metrics calculation later if needed
                        # Explicitly use `non_blocking=False` as this can cause
                        # race conditions in next accumulate
                        if use_cpu:
                            moved_report = report.copy().to("cpu", non_blocking=False)

                        # accumulate necessary params for metric calculation
                        if combined_report is None:
                            # make a copy of report since `reporter.add_to_report` will
                            # change some of the report keys later
                            combined_report = moved_report.copy()
                        else:
                            combined_report.accumulate_tensor_fields_and_loss(
                                moved_report, self.metrics.required_params
                            )
                            combined_report.batch_size += moved_report.batch_size

                        # Each node generates a separate copy of predict JSON from the
                        # report, which will be used to evaluate dataset-level metrics
                        # (such as mAP in object detection or CIDEr in image captioning)
                        # Since `reporter.add_to_report` changes report keys,
                        # (e.g scores) do this after
                        # `combined_report.accumulate_tensor_fields_and_loss`
                        if "__prediction_report__" in self.metrics.required_params:
                            # Still need to use original report here on GPU/TPU since
                            # it will be gathered
                            reporter.add_to_report(report, self.model)

                        if single_batch is True:
                            break

                logger.info("Averged time cost: {}".format(total_time_cost / total_sample))
                logger.info("TRT accuracy: {}".format(float(correct_prediction/total_sample)))
                logger.info(f"Finished training. Loaded {loaded_batches}")
                logger.info(f" -- skipped {skipped_batches} batches.")

                reporter.postprocess_dataset_report()
                assert (
                    combined_report is not None
                ), "Please check if your validation set is empty!"
                # add prediction_report is used for set-level metrics
                combined_report.prediction_report = reporter.report

                combined_report.metrics = self.metrics(combined_report, combined_report)

                # Since update_meter will reduce the metrics over GPUs, we need to
                # move them back to GPU but we will only move metrics and losses
                # which are needed by update_meter to avoid OOM
                # Furthermore, do it in a non_blocking way to avoid any issues
                # in device to host or host to device transfer
                if use_cpu:
                    combined_report = combined_report.to(
                        self.device, fields=["metrics", "losses"], non_blocking=False
                    )

                meter.update_from_report(combined_report, should_update_loss=False)

            # enable train mode again
            self.model.train()

        return combined_report, meter


    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        skipped_batches = 0
        loaded_batches = 0
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()
                if self._can_use_tqdm(dataloader):
                    dataloader = tqdm.tqdm(dataloader)

                for batch in dataloader:
                    # Do not timeout quickly on first batch, as workers might start at
                    # very different times.
                    with CompleteInTimeOrDie(600 if loaded_batches else 3600 * 24):
                        prepared_batch = reporter.prepare_batch(batch)
                        prepared_batch = to_device(prepared_batch, self.device)
                        loaded_batches += 1
                        if not validate_batch_sizes(prepared_batch.get_batch_size()):
                            logger.info("Skip batch due to unequal batch sizes.")
                            skipped_batches += 1
                            continue
                        with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                            model_output = self.model(prepared_batch)
                        report = Report(prepared_batch, model_output)
                        reporter.add_to_report(report, self.model)
                        report.detach()

                reporter.postprocess_dataset_report()

            logger.info(f"Finished predicting. Loaded {loaded_batches}")
            logger.info(f" -- skipped {skipped_batches} batches.")
            self.model.train()

    def _can_use_tqdm(self, dataloader: torch.utils.data.DataLoader):
        """
        Checks whether tqdm can be gracefully used with a dataloader
        1) should have `__len__` property defined
        2) calling len(x) should not throw errors.
        """
        use_tqdm = hasattr(dataloader, "__len__")

        try:
            _ = len(dataloader)
        except (AttributeError, TypeError, NotImplementedError):
            use_tqdm = False
        return use_tqdm


def validate_batch_sizes(my_batch_size: int) -> bool:
    """
    Validates all workers got the same batch size.
    """

    # skip batch size validation on XLA (as there's too much overhead
    # and data loader automatically drops the last batch in XLA mode)
    if is_xla():
        return True

    batch_size_tensor = torch.IntTensor([my_batch_size])
    if torch.cuda.is_available():
        batch_size_tensor = batch_size_tensor.cuda()
    all_batch_sizes = gather_tensor(batch_size_tensor)
    for j, oth_batch_size in enumerate(all_batch_sizes.data):
        if oth_batch_size != my_batch_size:
            logger.error(f"Node {j} batch {oth_batch_size} != {my_batch_size}")
            return False
    return True
