"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
import librosa
import numpy as np

@registry.register_processor("blap_audio_train")
@registry.register_processor("blap_audio_eval")
# TODO: 对clotho数据集中的wav文件进行预处理
class BlapAudioProcessor(BaseProcessor):
    def __init__(self, sample_rate, max_sec):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_sec = max_sec

    def __call__(self, item):
        y, _ = librosa.load(item, sr = self.sample_rate)
        y = self.pad_truncate_sequence(y, self.sample_rate * self.max_sec)
        y = y.astype(np.float32)
        return y
    
    def pad_truncate_sequence(self, y, max_len):
        if len(y) < max_len:
            repeat_cnt = max_len // len(y)
            y = np.tile(y, repeat_cnt)
            if len(y) < max_len:
                y = np.concatenate((y, y[0 : max_len - len(y)]))
            return y
        else:
            return y[0 : max_len]
        
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        sample_rate = cfg.get("sample_rate", 32000)
        max_sec = cfg.get("max_sec", 30)

        return cls(
            sample_rate = sample_rate,
            max_sec = max_sec
        )

@registry.register_processor("blap_caption")
# TODO: 对clotho数据集中的caption进行预处理
class BlapCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption