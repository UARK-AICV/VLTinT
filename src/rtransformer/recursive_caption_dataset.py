import copy
import torch
import logging
import math
import nltk
import numpy as np
import os
import pickle

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from src.utils import load_json, flat_list_of_lists

log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def load_c3d_env(path, dset_name="anet"):
    feat_dict = load_json(path)
    if dset_name == "anet":
        feature = np.array([d["features"][0] for d in feat_dict["video_features"]]) # anet
    elif dset_name == "yc2":
        feature = np.array([d["features"] for d in feat_dict["video_features"]]) # yc2
    return feature


def load_c3d_agent(path):
    feat_dict = load_json(path)
    feature = [np.array(d["features"]) if len(d['features'])>0 else np.empty((0,2048)) for i, d in enumerate(feat_dict["video_features"])]
    return feature
    

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


class RecursiveCaptionDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+text joint sequence
    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"
    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    IGNORE = -1  # used to calculate loss

    data_path = {
        "anet": {
            "env_feat": "data/anet/anet_env_c3d",
            "agent_feat": "/home/khoavo/datasets/Anet-1.3/c3d_agent_features/",
            "clip_feat": "/home/kashu/research/VLRT/anet_clip_b16/lang_feature/",
            "sent_feat": "/home/kashu/research/VLRT/anet_clip_b16/sent_feature/"
        },
        "yc2": {
            "env_feat": "/home/kashu/research/VLRT/yc2_c3d_env_features/",
            "agent_feat": "/home/kashu/research/VLRT/yc2_agent_features/",
            "clip_feat": "/home/kashu/research/VLRT/yc2_clip_b16/lang_feature/",
            "sent_feat": "/home/kashu/research/VLRT/yc2_clip_b16/sent_feature/"
        }
    }

    """
    recurrent: if True, return recurrent data
    """
    def __init__(self, dset_name, data_dir, duration_file, word2idx_path,
                 max_t_len, max_v_len, max_n_sen, mode="train", recurrent=True, untied=False, combine_image_feature=False):
        self.dset_name = dset_name
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.data_dir = data_dir  # containing training data
        self.env_feature_dir = self.data_path[dset_name]["env_feat"]
        self.agent_feature_dir = self.data_path[dset_name]["agent_feat"]
        self.clip_feature_dir = self.data_path[dset_name]["clip_feat"] 
        self.sent_feature_dir = self.data_path[dset_name]["sent_feat"]
        self.duration_file = duration_file
        # self.frame_to_second = self._load_duration()
        self._load_duration()
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sen
        self.max_n_sen = max_n_sen
        self.combine_image_feature = combine_image_feature

        self.mode = mode
        self.recurrent = recurrent
        self.untied = untied
        assert not (self.recurrent and self.untied), "untied and recurrent cannot be True for both"

        # data entries
        self.data = None
        self.set_data_mode(mode=mode)
        self.missing_video_names = []
        self.fix_missing()

        self.num_sens = None  # number of sentence for each video, set in self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta

    def set_data_mode(self, mode):
        """mode: `train` or `val`"""
        logging.info("Mode {}".format(mode))
        self.mode = mode
        if self.dset_name == "anet":
            if mode == "train":  # 10000 videos
                data_path = os.path.join(self.data_dir, "train.json")
            elif mode == "val":  # 2500 videos
                data_path = os.path.join(self.data_dir, "anet_entities_val_1.json")
            elif mode == "test":  # 2500 videos
                data_path = os.path.join(self.data_dir, "anet_entities_test_1.json")
            else:
                raise ValueError("Expecting mode to be one of [`train`, `val`, `test`], got {}".format(mode))
        elif self.dset_name == "yc2":
            if mode == "train":  # 10000 videos
                data_path = os.path.join(self.data_dir, "yc2_train_anet_format.json")
            elif mode == "val":  # 2500 videos
                data_path = os.path.join(self.data_dir, "yc2_val_anet_format.json")
            else:
                raise ValueError("Expecting mode to be one of [`train`, `val`, `test`], got {}".format(mode))
        else:
            raise ValueError
        self._load_data(data_path)

    def fix_missing(self):
        """filter our videos with no feature file"""
        for e in tqdm(self.data):
            video_name = e["name"][2:] if self.dset_name == "anet" else e["name"]
            feat_path_env = os.path.join(self.env_feature_dir, "v_{}.json".format(video_name) if self.dset_name == "anet" else "{}.json".format(video_name))
            feat_path_agent = os.path.join(self.agent_feature_dir, "v_{}.json".format(video_name) if self.dset_name == "anet" else "{}.json".format(video_name))
            feat_path_clip = os.path.join(self.clip_feature_dir, "v_{}.json".format(video_name) if self.dset_name == "anet" else "{}.json".format(video_name))
            if video_name not in self.duration.keys():
                self.missing_video_names.append(video_name)
            
            for p in [feat_path_env, feat_path_agent, feat_path_clip]:
                if not os.path.exists(p):
                    self.missing_video_names.append(video_name)
        print("Missing {} features (clips/sentences) from {} videos".format(
            len(self.missing_video_names), len(set(self.missing_video_names))))
        print("Missing {}".format(set(self.missing_video_names)))
        if self.dset_name == "anet":
            self.data = [e for e in self.data if e["name"][2:] not in self.missing_video_names]
        else:
            self.data = [e for e in self.data if e["name"] not in self.missing_video_names]

    def _load_duration(self):
        """https://github.com/salesforce/densecap/blob/master/data/anet_dataset.py#L120
        Since the features are extracted not at the exact 0.5 secs. To get the real time for each feature,
        use `(idx + 1) * frame_to_second[vid_name] `
        """
        # sampling_sec = 0.5  # hard coded, only support 0.5
        duration = {}
        if self.dset_name == "anet":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    duration[vid_name] = float(vid_dur)
                # frame_to_second["_0CqozZun3U"] = sampling_sec  # a missing video in anet
                duration["_0CqozZun3U"] = 294.227 # 8818 frames
        elif self.dset_name == "yc2":
            with open(self.duration_file, "r") as f:
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(",")]
                    duration[vid_name] = float(vid_dur)
        else:
            raise NotImplementedError("Only support anet and yc2, got {}".format(self.dset_name))

        self.duration = duration

    def _load_data(self, data_path):
        logging.info("Loading data from {}".format(data_path))
        raw_data = load_json(data_path)
        data = []
        for k, line in tqdm(raw_data.items()):
            line["name"] = k
            line["timestamps"] = line["timestamps"][:self.max_n_sen]
            line["sentences"] = line["sentences"][:self.max_n_sen]
            data.append(line)

        if self.recurrent:  # recurrent
            self.data = data
        else:  # non-recurrent single sentence
            singel_sentence_data = []
            for d in data:
                num_sen = min(self.max_n_sen, len(d["sentences"]))
                singel_sentence_data.extend([
                    {
                        "duration": d["duration"],
                        "name": d["name"],
                        "timestamp": d["timestamps"][idx],
                        "sentence": d["sentences"][idx]
                    } for idx in range(num_sen)])
            self.data = singel_sentence_data

        logging.info("Loading complete! {} examples".format(len(self)))

    def convert_example_to_features(self, example):
        """example single snetence
        {"name": str,
         "duration": float,
         "timestamp": [st(float), ed(float)],
         "sentence": str
        } or
        {"name": str,
         "duration": float,
         "timestamps": list([st(float), ed(float)]),
         "sentences": list(str)
        }
        """
        name = example["name"]
        video_name = name[2:] if self.dset_name == "anet" else name
        
        feat_path_env = os.path.join(self.env_feature_dir, "v_{}.json".format(video_name) if self.dset_name == "anet" else "{}.json".format(video_name))
        feat_path_agent = os.path.join(self.agent_feature_dir, "v_{}.json".format(video_name) if self.dset_name == "anet" else "{}.json".format(video_name))
        feat_path_clip = os.path.join(self.clip_feature_dir, "v_{}.json".format(video_name) if self.dset_name == "anet" else "{}.json".format(video_name))
        video_feature = load_c3d_env(feat_path_env, self.dset_name)
        agent_feature = load_c3d_agent(feat_path_agent)
        clip_feature = load_json(feat_path_clip)

        if self.mode == "train":
            sent_feat = load_json(os.path.join(self.sent_feature_dir, "v_{}.json".format(video_name) if self.dset_name == "anet" else "{}.json".format(video_name)))
            sent_feat = np.array(sent_feat)

        if self.recurrent:  # recurrent
            num_sen = len(example["sentences"])
            single_video_features = []
            single_video_meta = []
            for clip_idx in range(num_sen):
                cur_data, cur_meta = self.clip_sentence_to_feature(example["name"],
                                                                   example["timestamps"][clip_idx],
                                                                   example["sentences"][clip_idx],
                                                                   video_feature,
                                                                   agent_feature,
                                                                   clip_feature)
                if self.mode == "train":
                    cur_data["sent_feat"] = sent_feat[clip_idx].astype(np.float32)
                single_video_features.append(cur_data)
                single_video_meta.append(cur_meta)
            return single_video_features, single_video_meta
        else:  # single sentence
            clip_dataloader = self.clip_sentence_to_feature_untied \
                if self.untied else self.clip_sentence_to_feature
            cur_data, cur_meta = clip_dataloader(example["name"],
                                                 example["timestamp"],
                                                 example["sentence"],
                                                 video_feature)
            return cur_data, cur_meta

    def clip_sentence_to_feature(self, name, timestamp, sentence, video_feature, agent_feature, lang_features):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            video_feature: np array
        """
        # frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]
        duration = self.duration[name[2:]] if self.dset_name == "anet" else self.duration[name]

        # video + text tokens
        feat_video, video_tokens, video_mask, feat_agent, agent_mask, feat_lang, lang_mask = self._load_indexed_video_feature(
            video_feature, lang_features, agent_feature, timestamp, duration)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        input_tokens = video_tokens + text_tokens

        input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in input_tokens]
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        input_labels = \
            [self.IGNORE] * len(video_tokens) + \
            [self.IGNORE if m == 0 else tid for tid, m in zip(input_ids[-len(text_mask):], text_mask)][1:] + \
            [self.IGNORE]

        input_mask = video_mask + text_mask
        token_type_ids = [0] * self.max_v_len + [1] * self.max_t_len
        
        data = dict(
            name=name,
            # model inputs
            input_ids=np.array(input_ids).astype(np.int64),
            input_labels=np.array(input_labels).astype(np.int64),
            input_mask=np.array(input_mask).astype(np.float32),
            token_type_ids=np.array(token_type_ids).astype(np.int64),
            video_feature=feat_video.astype(np.float32),
            agent_feature=feat_agent.astype(np.float32), 
            agent_mask=agent_mask.astype(np.float32),
            lang_feature=feat_lang.astype(np.int64),
            lang_mask=lang_mask.astype(np.float32),
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamp,
            sentence=sentence,
        )
        return data, meta

    def clip_sentence_to_feature_untied(self, name, timestamp, sentence, raw_video_feature):
        """ make features for a single clip-sentence pair.
        [CLS], [VID], ..., [VID], [SEP], [BOS], [WORD], ..., [WORD], [EOS]
        Args:
            name: str,
            timestamp: [float, float]
            sentence: str
            raw_video_feature: np array, N x D, for the whole video
        """
        # frm2sec = self.frame_to_second[name[2:]] if self.dset_name == "anet" else self.frame_to_second[name]
        duration = self.duration[name[2:]] if self.dset_name == "anet" else self.duration[name]

        # video + text tokens
        video_feature, video_mask = self._load_indexed_video_feature_untied(raw_video_feature, timestamp, duration)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        text_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in text_tokens]
        # shifted right, `-1` is ignored when calculating CrossEntropy Loss
        text_labels = [self.IGNORE if m == 0 else tid for tid, m in zip(text_ids, text_mask)][1:] + [self.IGNORE]

        data = dict(
            name=name,
            text_tokens=text_tokens,
            # model inputs
            text_ids=np.array(text_ids).astype(np.int64),
            text_mask=np.array(text_mask).astype(np.float32),
            text_labels=np.array(text_labels).astype(np.int64),
            video_feature=video_feature.astype(np.float32),
            video_mask=np.array(video_mask).astype(np.float32),
        )
        meta = dict(
            # meta
            name=name,
            timestamp=timestamp,
            sentence=sentence,
        )
        return data, meta

    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, duration, total_frames):
        """convert wall time st_ed to feature index st_ed"""
        st = int(math.floor((timestamp[0] / duration) * total_frames))
        ed = int(math.ceil((timestamp[1] / duration) * total_frames))
        ed = min(ed, feat_len-1)
        st = min(st, ed-1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed

    def _load_indexed_video_feature(self, raw_vid_feat, raw_lang_feat, raw_agent_feature, timestamp, duration):
        """ [CLS], [VID], ..., [VID], [SEP], [PAD], ..., [PAD],
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat is padded to length of (self.max_v_len + self.max_t_len,)
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        max_v_l = self.max_v_len - 2
        num_agent = 35
        feat_video = np.zeros((self.max_v_len, raw_vid_feat.shape[1]))  # includes [CLS], [SEP]
        # feat_image = np.zeros((self.max_v_len, raw_image_feat.shape[1]))
        feat_agent = np.zeros((self.max_v_len, num_agent, raw_vid_feat.shape[1]))
        agent_mask = np.zeros((self.max_v_len, num_agent))
        feat_lang = np.zeros((self.max_v_len, 35))
        lang_mask = np.zeros((self.max_v_len, 35))

        # load video feature
        feat_len = len(raw_vid_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, duration, feat_len)
        indexed_feat_len = ed - st + 1
        valid_l = max_v_l
        if indexed_feat_len > max_v_l:
            downsample_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsample_indices) < feat_len
            feat_video[1:max_v_l+1] = raw_vid_feat[downsample_indices]  # truncate, sample???
            for i, d in enumerate(downsample_indices, 1):
                feat_agent[i,:len(raw_agent_feature[d])] = raw_agent_feature[d][:num_agent] 
                agent_mask[i,:len(raw_agent_feature[d])] = 1

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            video_mask = [1] * (max_v_l + 2)

        else:
            valid_l = ed - st + 1
            feat_video[1:valid_l+1] = raw_vid_feat[st:ed + 1]
            for i, d in enumerate(range(st, ed+1), 1):
                feat_agent[i,:len(raw_agent_feature[d])] = raw_agent_feature[d][:num_agent] 
                agent_mask[i,:len(raw_agent_feature[d])] = 1

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + \
                           [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
            video_mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)

        # load image feature
        feat_len = len(raw_lang_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, duration, feat_len)
        tmp = ed - st + 1
        downsample_indices = np.linspace(st, ed, tmp, endpoint=True).astype(np.int)
        downsample_downsample_indices = np.linspace(0, tmp - 1, valid_l, endpoint=True).astype(np.int).tolist()
        downsample_indices = downsample_indices[downsample_downsample_indices].tolist()

        for i, d in enumerate(downsample_indices, 1):
            feat_lang[i] =  np.array([self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in raw_lang_feat[d][:35]])
            lang_mask[i] = 1

        return feat_video, video_tokens, video_mask, feat_agent, agent_mask, feat_lang, lang_mask

    def _load_indexed_video_feature_untied(self, raw_feat, timestamp, duration):
        """ Untied version: [VID], ..., [VID], [PAD], ..., [PAD], len == max_v_len
        Returns:
            feat is padded to length of (self.max_v_len,)
            mask: self.max_v_len, with 1 indicates valid bits, 0 indicates padding
        """
        max_v_l = self.max_v_len
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, duration, feat_len)
        indexed_feat_len = ed - st + 1

        if indexed_feat_len > max_v_l:
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat = raw_feat[downsamlp_indices]  # truncate, sample???
            mask = [1] * max_v_l  # no padding
        else:
            feat = np.zeros((max_v_l, raw_feat.shape[1]))  # only video features and padding
            valid_l = ed - st + 1
            feat[:valid_l] = raw_feat[st:ed + 1]
            mask = [1] * valid_l + [0] * (max_v_l - valid_l)
        return feat, mask

    def _tokenize_pad_sentence(self, sentence):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


def prepare_batch_inputs(batch, device, non_blocking=False):
    batch_inputs = dict()
    bsz = len(batch["name"])
    for k, v in batch.items():
        assert bsz == len(v), (bsz, k, v)
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        else:  # all non-tensor values
            batch_inputs[k] = v
    return batch_inputs


def step_collate(padded_batch_step):
    """The same step (clip-sentence pair) from each example"""
    c_batch = dict()
    for key in padded_batch_step[0]:
        value = padded_batch_step[0][key]
        if isinstance(value, list):
            c_batch[key] = [d[key] for d in padded_batch_step]
        else:
            c_batch[key] = default_collate([d[key] for d in padded_batch_step])
    return c_batch


def caption_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66

    HOW to batch clip-sentence pair?
    1) directly copy the last sentence, but do not count them in when back-prop OR
    2) put all -1 to their text token label, treat
    """
    # collect meta
    raw_batch_meta = [e[1] for e in batch]
    batch_meta = []
    for e in raw_batch_meta:
        cur_meta = dict(
            name=None,
            timestamp=[],
            gt_sentence=[]
        )
        for d in e:
            cur_meta["name"] = d["name"]
            cur_meta["timestamp"].append(d["timestamp"])
            cur_meta["gt_sentence"].append(d["sentence"])
        batch_meta.append(cur_meta)

    batch = [e[0] for e in batch]
    # Step1: pad each example to max_n_sen
    max_n_sen = max([len(e) for e in batch])
    raw_step_sizes = []

    padded_batch = []
    padding_clip_sen_data = copy.deepcopy(batch[0][0])  # doesn"t matter which one is used
    padding_clip_sen_data["input_labels"][:] = RecursiveCaptionDataset.IGNORE
    for ele in batch:
        cur_n_sen = len(ele)
        if cur_n_sen < max_n_sen:
            ele = ele + [padding_clip_sen_data] * (max_n_sen - cur_n_sen)
        raw_step_sizes.append(cur_n_sen)
        padded_batch.append(ele)

    # Step2: batching each steps individually in the batches
    collated_step_batch = []
    for step_idx in range(max_n_sen):
        collated_step = step_collate([e[step_idx] for e in padded_batch])
        collated_step_batch.append(collated_step)
    return collated_step_batch, raw_step_sizes, batch_meta


def single_sentence_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    # collect meta
    batch_meta = [{"name": e[1]["name"],
                   "timestamp": e[1]["timestamp"],
                   "gt_sentence": e[1]["sentence"]
                   } for e in batch]  # change key
    padded_batch = step_collate([e[0] for e in batch])
    return padded_batch, None, batch_meta
