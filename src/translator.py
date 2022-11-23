""" This module will handle the text generation with beam search. """

import torch
import copy
import torch.nn.functional as F
from src.rtransformer.model import RecursiveTransformer
from src.rtransformer.recursive_caption_dataset import RecursiveCaptionDataset as RCDataset

import logging
logger = logging.getLogger(__name__)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def mask_tokens_after_eos(input_ids, input_masks,
                          eos_token_id=RCDataset.EOS, pad_token_id=RCDataset.PAD):
    """replace values after `[EOS]` with `[PAD]`,
    used to compute memory for next sentence generation"""
    for row_idx in range(len(input_ids)):
        # possibly more than one `[EOS]`
        cur_eos_idxs = (input_ids[row_idx] == eos_token_id).nonzero()
        if len(cur_eos_idxs) != 0:
            cur_eos_idx = cur_eos_idxs[0, 0].item()
            input_ids[row_idx, cur_eos_idx+1:] = pad_token_id
            input_masks[row_idx, cur_eos_idx+1:] = 0
    return input_ids, input_masks


class Translator(object):
    """Load with trained model and handle the beam search"""
    def __init__(self, opt, checkpoint, model=None):
        self.opt = opt
        self.device = torch.device("cuda" if opt.cuda else "cpu")

        self.model_config = checkpoint["model_cfg"]
        self.max_t_len = self.model_config.max_t_len
        self.max_v_len = self.model_config.max_v_len
        self.num_hidden_layers = self.model_config.num_hidden_layers

        if model is None:
            logger.info("Use recurrent model - Mine")
            model = RecursiveTransformer(self.model_config).to(self.device)
            model.load_state_dict(checkpoint["model"])

        print("[Info] Trained model state loaded.")
        self.model = model
        self.model.eval()

    def translate_batch_greedy(self, 
        input_ids_list,
        video_features_list,
        input_masks_list,
        token_type_ids_list,
        agent_feature_list,
        agent_mask_list,
        lang_feature_list,
        lang_mask_list,
        input_labels_list, rt_model):
        def greedy_decoding_step(prev_ms, input_ids, video_features,
                            input_masks, token_type_ids, agent_feature,
                            agent_mask, lang_feature, lang_mask,
                            model, max_v_len, max_t_len, start_idx=RCDataset.BOS, unk_idx=RCDataset.UNK):
            """RTransformer The first few args are the same to the input to the forward_step func
            Note:
                1, Copy the prev_ms each word generation step, as the func will modify this value,
                which will cause discrepancy between training and inference
                2, After finish the current sentence generation step, replace the words generated
                after the `[EOS]` token with `[PAD]`. The replaced input_ids should be used to generate
                next memory state tensor.
            """
            bsz = len(input_ids)
            next_symbols = torch.LongTensor([start_idx] * bsz)  # (N, )
            for dec_idx in range(max_v_len, max_v_len + max_t_len):
                input_ids[:, dec_idx] = next_symbols
                input_masks[:, dec_idx] = 1
                
                copied_prev_ms = copy.deepcopy(prev_ms)  # since the func is changing data inside
                _, _, pred_scores = model.forward_step(
                    copied_prev_ms, input_ids, video_features,
                    input_masks, token_type_ids, agent_feature,
                    agent_mask, lang_feature, lang_mask
                )
                # suppress unk token; (N, L, vocab_size)
                pred_scores[:, :, unk_idx] = -1e10
                # next_words = pred_scores.max(2)[1][:, dec_idx]
                next_words = pred_scores[:, dec_idx].max(1)[1]  # TODO / NOTE changed
                next_symbols = next_words

            # compute memory, mimic the way memory is generated at training time
            input_ids, input_masks = mask_tokens_after_eos(input_ids, input_masks)
            cur_ms, _, pred_scores = model.forward_step(prev_ms, input_ids, video_features,
                    input_masks, token_type_ids, agent_feature,
                    agent_mask, lang_feature, lang_mask)

            return cur_ms, input_ids[:, max_v_len:]  # (N, max_t_len == L-max_v_len)

        input_ids_list, input_masks_list = self.prepare_video_only_inputs(
            input_ids_list, input_masks_list, token_type_ids_list)
        for cur_input_masks in input_ids_list:
            assert torch.sum(cur_input_masks[:, self.max_v_len + 1:]) == 0, \
                "Initially, all text tokens should be masked"

        config = rt_model.config
        with torch.no_grad():
            prev_ms = [None] * config.num_hidden_layers
            step_size = len(input_ids_list)
            dec_seq_list = []
            for idx in range(step_size):
                prev_ms, dec_seq = greedy_decoding_step(
                    prev_ms, input_ids_list[idx], video_features_list[idx],
                    input_masks_list[idx], token_type_ids_list[idx],
                    agent_feature_list[idx],
                    agent_mask_list[idx],
                    lang_feature_list[idx],
                    lang_mask_list[idx],
                    rt_model, config.max_v_len, config.max_t_len)
                dec_seq_list.append(dec_seq)
            return dec_seq_list


    def translate_batch(self, model_inputs, use_beam=False, recurrent=True, untied=False, xl=False, mtrans=False, mmt2=True):
        """while we used *_list as the input names, they could be non-list for single sentence decoding case"""
        return self.translate_batch_greedy(
            *model_inputs, self.model)

    @classmethod
    def prepare_video_only_inputs(cls, input_ids, input_masks, segment_ids):
        """ replace text_ids (except `[BOS]`) in input_ids with `[PAD]` token, for decoding.
        This function is essential!!!
        Args:
            input_ids: (N, L) or [(N, L)] * step_size
            input_masks: (N, L) or [(N, L)] * step_size
            segment_ids: (N, L) or [(N, L)] * step_size
        """
        if isinstance(input_ids, list):
            video_only_input_ids_list = []
            video_only_input_masks_list = []
            for e1, e2, e3 in zip(input_ids, input_masks, segment_ids):
                text_mask = e3 == 1  # text positions (`1`) are replaced
                e1[text_mask] = RCDataset.PAD
                e2[text_mask] = 0  # mark as invalid bits
                video_only_input_ids_list.append(e1)
                video_only_input_masks_list.append(e2)
            return video_only_input_ids_list, video_only_input_masks_list
        else:
            text_mask = segment_ids == 1
            input_ids[text_mask] = RCDataset.PAD
            input_masks[text_mask] = 0
            return input_ids, input_masks
