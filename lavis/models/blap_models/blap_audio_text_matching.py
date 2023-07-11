"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn.functional as F
from lavis.common.registry import registry
# from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blap_models.blap_qformer import BlapQformer


@registry.register_model("blap_audio_text_matching")
class BlapITM(BlapQformer):
    """
    BLAP Audio-Text Matching (ITM) model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_image_text_matching", "pretrained")
        >>> model = load_model("blip2_image_text_matching", "coco")
    """

    def __init__(
        self,
        # vit_model="eva_clip_g",
        # img_size=224,
        # drop_path_rate=0,
        # use_grad_checkpoint=False,
        # vit_precision="fp16",
        freeze_audio=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
    ):
        super().__init__(
            # vit_model=vit_model,
            # img_size=img_size,
            # drop_path_rate=drop_path_rate,
            # use_grad_checkpoint=use_grad_checkpoint,
            # vit_precision=vit_precision,
            freeze_audio=freeze_audio,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )

    def forward(self, samples, match_head="itm"):
        waveform = samples["waveform"]
        caption = samples["text_input"]

        with self.maybe_autocast():
            audio_output_dict = self.audio_encoder(waveform)
            audio_embeds = self.ln_audio(audio_output_dict['latent_output'])
        audio_embeds = audio_embeds.float()
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(
            waveform.device
        )

        text = self.tokenizer(
            caption,
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(waveform.device)

        if match_head == "itm":
            query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                waveform.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            output_itm = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=audio_embeds,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )
            itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
            itm_logit = self.itm_head(itm_embeddings)
            itm_logit = itm_logit.mean(dim=1)

            return itm_logit

        elif match_head == "itc":
            query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=audio_embeds,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )
            audio_feats = F.normalize(
                self.audio_proj(query_output.last_hidden_state), dim=-1
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sims = torch.bmm(audio_feats, text_feat.unsqueeze(-1))
            sim, _ = torch.max(sims, dim=1)

            return sim
