"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import warnings
import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
# from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset
from lavis.datasets.datasets.clotho_caption_datasets import ClothoCaptionDataset, ClothoCapEvalDataset
# from lavis.datasets.datasets.laion_dataset import LaionDataset
from lavis.processors.base_processor import BaseProcessor

import torch.distributed as dist
from lavis.common.dist_utils import is_dist_avail_and_initialized
import lavis.common.utils as utils

@registry.register_builder("clotho_caption")
class ClothoCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClothoCaptionDataset
    eval_dataset_cls = ClothoCapEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/clotho/defaults_caption.yaml"}

    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)

        self.audio_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        # don't download, split, etc...
        # only called on 1 GPU/TPU in distributed

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        audio_proc_cfg = self.config.get("audio_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if audio_proc_cfg is not None:
            audio_train_cfg = audio_proc_cfg.get("train")
            audio_eval_cfg = audio_proc_cfg.get("eval")

            self.audio_processors["train"] = self._build_proc_from_cfg(audio_train_cfg)
            self.audio_processors["eval"] = self._build_proc_from_cfg(audio_eval_cfg)

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)

    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        audio_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            audio_processor = (
                self.audio_processors["train"]
                if is_train
                else self.audio_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )
            # choose one caption from five candidates
            which_caption = ann_info.get(split).which_caption

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            abs_ann_paths = []
            for ann_path in ann_paths:
                if not os.path.isabs(ann_path):
                    ann_path = utils.get_cache_path(ann_path)
                abs_ann_paths.append(ann_path)
            ann_paths = abs_ann_paths

            # audio data storage path
            audio_dir = audio_info.get(split).storage

            if not os.path.isabs(audio_dir):
                # vis_path = os.path.join(utils.get_cache_path(), vis_path)
                audio_dir = utils.get_cache_path(audio_dir)

            if not os.path.exists(audio_dir):
                warnings.warn("storage path {} does not exist.".format(audio_dir))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            datasets[split] = dataset_cls(
                audio_processor=audio_processor,
                text_processor=text_processor,
                audio_root=audio_dir,
                caption_paths=ann_paths,
                which_caption=which_caption
            )

        return datasets