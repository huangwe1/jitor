import jittor as jt
from jittor import nn
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import numpy as np
import json
import os
import re


class SimpleTokenizer:
    """Simple BERT-compatible tokenizer without requiring the transformers library."""
    def __init__(self, vocab_file=None, max_len=40):
        self.max_len = max_len
        self.vocab = {}
        self.pad_token_id = 0
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.unk_token_id = 100
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        else:
            self._build_basic_vocab()

    def _build_basic_vocab(self):
        self.vocab = {'[PAD]': 0, '[UNK]': 100, '[CLS]': 101, '[SEP]': 102, '[MASK]': 103}
        for i in range(30522):
            if i not in self.vocab.values():
                self.vocab[f'##unused{i}'] = i

    def _load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.vocab[token] = idx

    def tokenize(self, text):
        text = text.lower().strip()
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def encode(self, text, max_length=None):
        if max_length is None:
            max_length = self.max_len
        tokens = self.tokenize(text)
        ids = [self.cls_token_id] + self.convert_tokens_to_ids(tokens)[:max_length - 2] + [self.sep_token_id]
        attn_mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
            attn_mask.append(0)
        return ids[:max_length], attn_mask[:max_length]


try:
    from transformers import BertTokenizer
    _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    USE_BERT_TOKENIZER = True
except:
    _tokenizer = SimpleTokenizer()
    USE_BERT_TOKENIZER = False


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], jt.Var):
        return jt.stack(x)
    return x


class BaseProcessing:
    def __init__(self, transform=None, template_transform=None, search_transform=None, joint_transform=None):
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search': transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        jittered_size = box[2:4] * jt.exp(jt.array(np.random.randn(2).astype(np.float32)) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * jt.array([self.center_jitter_factor[mode]]).float32())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (jt.array(np.random.rand(2).astype(np.float32)) - 0.5)
        return jt.concat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            w, h = jt.stack(jittered_anno, dim=0)[:, 2], jt.stack(jittered_anno, dim=0)[:, 3]
            crop_sz = jt.ceil(jt.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                return data

            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    return data
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16
                ele_t = ele.float32().unsqueeze(0).unsqueeze(0) if isinstance(ele, jt.Var) else jt.array(ele.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                mask_down = nn.interpolate(ele_t, size=feat_size).bool()
                if mask_down.all():
                    data['valid'] = False
                    return data

        data['valid'] = True
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = jt.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = jt.zeros((1, self.output_sz["search"], self.output_sz["search"]))

        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        nlp_len = 40
        if data['language'] == [] or data['language'] is None:
            phrase_ids = jt.zeros(nlp_len, dtype=jt.int64)
            phrase_attnmask = jt.zeros(nlp_len, dtype=jt.int64)
        else:
            phrase = data['language']
            if isinstance(phrase, str):
                phrase = [phrase]

            if USE_BERT_TOKENIZER:
                encoded = _tokenizer.batch_encode_plus(phrase, padding='longest', return_tensors='np')
                ids_np = encoded['input_ids'].squeeze()
                mask_np = encoded['attention_mask'].squeeze()
                phrase_ids = jt.array(ids_np.astype(np.int64))
                phrase_attnmask = jt.array(mask_np.astype(np.int64))
            else:
                ids, mask = _tokenizer.encode(phrase[0], max_length=nlp_len)
                phrase_ids = jt.array(np.array(ids, dtype=np.int64))
                phrase_attnmask = jt.array(np.array(mask, dtype=np.int64))

            if phrase_ids.ndim == 1 and phrase_ids.shape[0] < nlp_len:
                pad = jt.zeros(nlp_len - phrase_ids.shape[0], dtype=jt.int64)
                phrase_ids = jt.concat([phrase_ids, pad], dim=0)
                phrase_attnmask = jt.concat([phrase_attnmask, pad], dim=0)
            elif phrase_ids.ndim == 1 and phrase_ids.shape[0] > nlp_len:
                phrase_ids = phrase_ids[:nlp_len]
                phrase_attnmask = phrase_attnmask[:nlp_len]

        data['phrase_ids'] = phrase_ids
        data['phrase_attnmask'] = phrase_attnmask
        return data
