"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
from typing import Any

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
import librosa
import numpy as np
import torch.nn as nn
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

class GenLogmel(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=1024, hop_length=320, 
            win_length=1024, window='hann', center=True, pad_mode='reflect', 
            freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=32000, n_fft=1024, 
            n_mels=64, fmin=50, fmax=14000, ref=1.0, amin=1e-10, top_db=None, 
            freeze_parameters=True)
        
        file_path = '/mnt/wjr/LAVIS/lavis/models/htsat_models/ckpt/gen_logmel.ckpt'
        state_dict = torch.load(file_path)

        # print("================", state_dict.keys(), "=====================")
        self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        if torch.any(torch.isnan(x)).item():
            print("true")

        return x
        


@registry.register_processor("blap_audio_train")
@registry.register_processor("blap_audio_eval")
# TODO: 对clotho数据集中的wav文件进行预处理
class BlapAudioProcessor(BaseProcessor):
    def __init__(self, sample_rate, max_sec):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_sec = max_sec

        self.gen_logmel = GenLogmel()

    def __call__(self, item):
        y, _ = librosa.load(item, sr = self.sample_rate)
        y = self.pad_truncate_sequence(y, self.sample_rate * self.max_sec)
        y = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)
        y = self.gen_logmel(y).squeeze(0)
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