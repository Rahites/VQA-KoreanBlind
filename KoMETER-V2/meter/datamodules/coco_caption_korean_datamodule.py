from ..datasets import CocoCaptionKoreanDataset
from .datamodule_base import BaseDataModule


class CocoCaptionKoreanDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CocoCaptionKoreanDataset

    @property
    def dataset_cls_no_false(self):
        return CocoCaptionKoreanDataset

    @property
    def dataset_name(self):
        return "coco_korean"
