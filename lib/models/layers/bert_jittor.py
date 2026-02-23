"""
Jittor implementation of BERT Embeddings (replacing transformers library).
Only implements BertConfig and BertEmbeddings as used in the All-in-One model.
"""
import jittor as jt
from jittor import nn


class BertConfig:
    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 max_position_embeddings=512,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 type_vocab_size=2,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_ids = jt.arange(config.max_position_embeddings).unsqueeze(0)
        self.token_type_ids = jt.zeros(self.position_ids.shape, dtype=jt.int64)

    def execute(self, input_ids=None, token_type_ids=None, position_ids=None):
        seq_length = input_ids.shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = jt.zeros(input_ids.shape, dtype=jt.int64)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
