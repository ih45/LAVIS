"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from torch.utils.data.dataloader import default_collate
from lavis.datasets.datasets.base_dataset import BaseDataset
# from PIL import Image
import pandas as pd

# TODO: 只适用于clotho
class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["file_name"]),
                "caption_1": ann["caption_1"],
                "caption_2": ann["caption_2"],
                "caption_3": ann["caption_3"],
                "caption_4": ann["caption_4"],
                "caption_5": ann["caption_5"],
                "waveform": sample["waveform"],
            }
        )


class AudioTextPairDataset(BaseDataset, __DisplMixin):
    def __init__(self, audio_processor, text_processor, audio_root, caption_paths, which_caption):
        """
        audio_root (string): Root directory of audios (e.g. /mnt1/datasets/Clotho/clotho_audio_development/development/)
        caption_path (string): annotation file path (.csv)
        """
        # TODO: 除text_processor外没用
        super().__init__(vis_processor=None, text_processor=text_processor, vis_root=None, ann_paths=[])

        self.audio_processor = audio_processor
        self.audio_root = audio_root
        self.caption_paths = caption_paths
        self.which_caption = which_caption # [caption_1|caption_2|caption_3|caption_4|caption_5]

        for caption_path in self.caption_paths:
            tmp_df = pd.read_csv(caption_path, encoding='utf-8')
            self.annotation.extend(tmp_df.to_dict('records'))
            # [{'file_name':, 'caption_1':, 'caption_2':, 'caption_3':, 'caption_4':, 'caption_5':}]

        self._add_instance_ids()

    def __getitem__(self, index):

        # TODO: 不保证适用于其他数据集
        ann = self.annotation[index]

        audio_path = os.path.join(self.audio_root, ann["file_name"])

        waveform = self.audio_processor(audio_path)
        caption = self.text_processor(ann[self.which_caption])

        return {"waveform": waveform, "text_input": caption}

    def set_processors(self, audio_processor, text_processor):
        self.audio_processor = audio_processor
        self.text_processor = text_processor

