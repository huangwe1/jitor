"""
Test script to verify the Jittor-converted All-in-One model can be constructed and run forward pass.
"""
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
args, _ = parser.parse_known_args()

# Use system CUDA 12.4 instead of Jittor's bundled CUDA 11.2
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
if os.path.exists(cuda_path) and not args.cpu:
    os.environ['nvcc_path'] = os.path.join(cuda_path, 'bin', 'nvcc.exe')
    os.environ['CUDA_PATH'] = cuda_path
elif args.cpu:
    os.environ['nvcc_path'] = ''

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jittor as jt

use_cuda = jt.has_cuda and not args.cpu
if use_cuda:
    jt.flags.use_cuda = 1
    print(f"Jittor version: {jt.__version__}")
    print("Using CUDA mode (GPU)")
else:
    jt.flags.use_cuda = 0
    print(f"Jittor version: {jt.__version__}")
    print("Using CPU mode")

print("\n=== Testing Model Components ===\n")

print("1. Testing PatchEmbed...")
from lib.models.layers.patch_embed import PatchEmbed
patch_embed = PatchEmbed(img_size=256, patch_size=16, in_chans=3, embed_dim=768)
x = jt.randn(2, 3, 256, 256)
out = patch_embed(x)
print(f"   Input: {x.shape} -> Output: {out.shape}")
assert out.shape == [2, 256, 768], f"Expected [2, 256, 768], got {out.shape}"
print("   PASSED!")

print("\n2. Testing Attention...")
from lib.models.layers.attn import Attention
attn = Attention(dim=768, num_heads=12, qkv_bias=True)
x = jt.randn(2, 320, 768)
out = attn(x)
print(f"   Input: {x.shape} -> Output: {out.shape}")
assert out.shape == [2, 320, 768], f"Expected [2, 320, 768], got {out.shape}"
print("   PASSED!")

print("\n3. Testing CEBlock...")
from lib.models.layers.attn_blocks import CEBlock
ce_block = CEBlock(dim=768, num_heads=12, qkv_bias=True, keep_ratio_search=0.7)
x = jt.randn(2, 320, 768)
global_index_t = jt.linspace(0, 63, 64).unsqueeze(0).repeat(2, 1)
global_index_s = jt.linspace(0, 255, 256).unsqueeze(0).repeat(2, 1)
out, gi_t, gi_s, removed, attn_map = ce_block(x, global_index_t, global_index_s)
print(f"   Input: {x.shape} -> Output: {out.shape}")
print(f"   Search tokens: {gi_s.shape[1]} (from {global_index_s.shape[1]})")
print("   PASSED!")

print("\n4. Testing CenterPredictor...")
from lib.models.layers.head import CenterPredictor
head = CenterPredictor(inplanes=768, channel=256, feat_sz=16, stride=16)
x = jt.randn(2, 768, 16, 16)
score_map_ctr, bbox, size_map, offset_map = head(x)
print(f"   Input: {x.shape}")
print(f"   Score map: {score_map_ctr.shape}, BBox: {bbox.shape}")
print("   PASSED!")

print("\n5. Testing BertEmbeddings...")
from lib.models.layers.bert_jittor import BertConfig, BertEmbeddings
bert_config = BertConfig(
    vocab_size=30522, hidden_size=768, num_hidden_layers=12,
    num_attention_heads=12, intermediate_size=48,
    max_position_embeddings=40, hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)
bert_emb = BertEmbeddings(bert_config)
input_ids = jt.randint(0, 30522, (2, 40))
out = bert_emb(input_ids)
print(f"   Input: {input_ids.shape} -> Output: {out.shape}")
assert out.shape == [2, 40, 768], f"Expected [2, 40, 768], got {out.shape}"
print("   PASSED!")

print("\n6. Testing VisionTransformerCE (backbone)...")
from lib.models.ostrack.vit_ce import vit_base_patch16_224_ce
backbone = vit_base_patch16_224_ce(
    pretrained=False,
    drop_path_rate=0.1,
    ce_loc=[3, 6, 9],
    ce_keep_ratio=[0.7, 0.7, 0.7]
)
from easydict import EasyDict
cfg = EasyDict({
    'DATA': {'SEARCH': {'SIZE': 256}, 'TEMPLATE': {'SIZE': 128}},
    'MODEL': {
        'BACKBONE': {'STRIDE': 16, 'CAT_MODE': 'direct', 'SEP_SEG': False},
        'RETURN_INTER': False, 'RETURN_STAGES': []
    }
})
backbone.finetune_track(cfg=cfg, patch_start_index=1)
print("   Backbone created and finetuned for tracking")

z = jt.randn(2, 3, 128, 128)
x = jt.randn(2, 3, 256, 256)
lang = jt.randn(2, 768, 1, 1)
ce_mask = jt.zeros([2, 8, 8])
ce_mask[:, 3:4, 3:4] = 1
ce_mask = ce_mask.flatten(1).bool()

out, aux_dict = backbone(z, x, lang, ce_template_mask=ce_mask, ce_keep_rate=None)
print(f"   Template: {z.shape}, Search: {x.shape}")
print(f"   Output: {out.shape}")
print(f"   Aux keys: {list(aux_dict.keys())}")
print("   PASSED!")

print("\n7. Testing full OSTrack model...")
from lib.models.ostrack.ostrack import OSTrack
from lib.models.layers.head import build_box_head

full_cfg = EasyDict({
    'DATA': {'SEARCH': {'SIZE': 256}, 'TEMPLATE': {'SIZE': 128}},
    'MODEL': {
        'PRETRAIN_FILE': '',
        'BACKBONE': {
            'TYPE': 'vit_base_patch16_224_ce',
            'STRIDE': 16, 'CAT_MODE': 'direct', 'SEP_SEG': False,
            'CE_LOC': [3, 6, 9], 'CE_KEEP_RATIO': [0.7, 0.7, 0.7]
        },
        'HEAD': {'TYPE': 'CENTER', 'NUM_CHANNELS': 256},
        'RETURN_INTER': False, 'RETURN_STAGES': [],
        'HIDDEN_DIM': 768,
    },
    'TRAIN': {'DROP_PATH_RATE': 0.1}
})

backbone2 = vit_base_patch16_224_ce(
    pretrained=False, drop_path_rate=0.1,
    ce_loc=[3, 6, 9], ce_keep_ratio=[0.7, 0.7, 0.7]
)
backbone2.finetune_track(cfg=full_cfg, patch_start_index=1)
box_head = build_box_head(full_cfg, 768)
model = OSTrack(backbone2, box_head, aux_loss=False, head_type="CENTER")

template = jt.randn(2, 3, 128, 128)
search = jt.randn(2, 3, 256, 256)
phrase_ids = jt.randint(0, 30522, (2, 40))
phrase_attnmask = jt.ones((2, 40)).int64()
ce_mask2 = jt.zeros([2, 8, 8])
ce_mask2[:, 3:4, 3:4] = 1
ce_mask2 = ce_mask2.flatten(1).bool()

output = model(template, search, phrase_ids, phrase_attnmask, ce_template_mask=ce_mask2)
print(f"   Template: {template.shape}, Search: {search.shape}")
print(f"   Pred boxes: {output['pred_boxes'].shape}")
print(f"   Score map: {output['score_map'].shape}")
print(f"   Output keys: {[k for k in output.keys() if k not in ['attn', 'removed_indexes_s', 'backbone_feat']]}")
print("   PASSED!")

print("\n" + "=" * 50)
print("ALL TESTS PASSED! Model converted successfully.")
print("=" * 50)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal model parameters: {total_params:,}")
