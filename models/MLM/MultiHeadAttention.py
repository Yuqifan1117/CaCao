import torch
from torch import nn
import math
from transformers.activations import GELUActivation

from models.MLM.utils import layer_init
class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        layer_init(self.query, xavier=True)
        layer_init(self.key, xavier=True)
        layer_init(self.value, xavier=True)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # attention_probs.shape = torch.Size([1, 12, 50, 50])

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

class FeedForwardNetwork(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.activate = GELUActivation()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        layer_init(self.dense1, xavier=True)
        layer_init(self.dense2, xavier=True)
    # dont use residual connection to transfer more information
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activate(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)


        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, intermediate_size, hidden_dropout_prob, layer_norm_eps):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = FeedForwardNetwork(intermediate_size, hidden_size, hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm(attention_output)
        outputs = self.output(layer_output)


        return outputs
