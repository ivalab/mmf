# Georgia Tech Robotic Task Reasoning
#
from mmf.common.registry import registry
from mmf.datasets.builders.sgl.dataset import SGLDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder

@registry.register_builder("sgl")
class SGLBuilder(MMFDatasetBuilder):
    def __init__(self, dataset_name="sgl", dataset_class=SGLDataset, *args, **kwargs):
        # Init should call super().__init__ with the key for the dataset
        super().__init__(dataset_name, dataset_class)

        # Assign the dataset class
        self.dataset_class = SGLDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/sgl/defaults.yaml"

    def load(self, *args, **kwargs):
        dataset = super().load(*args, **kwargs)
        if dataset is not None and hasattr(dataset, "try_fast_read"):
            dataset.try_fast_read()

        return dataset

    # TODO: Deprecate this method and move configuration updates directly to processors
    def update_registry_for_model(self, config):
        # Register both vocab (question and answer) sizes to registry for easy access to the
        # models. update_registry_for_model function if present is automatically called by
        # MMF
        registry.register(
            self.dataset_name + "_text_vocab_size",
            self.dataset.text_processor.get_vocab_size(),
        )
        registry.register(
            self.dataset_name + "_num_final_outputs",
            self.dataset.answer_processor.get_vocab_size(),
        )

@registry.register_builder("sgl_train_val")
class SGLTrainValBuilder(SGLBuilder):
    def __init__(self, dataset_name="sgl_train_val"):
        super().__init__(dataset_name)

    @classmethod
    def config_path(self):
        return "configs/datasets/sgl/train_val.yaml"
