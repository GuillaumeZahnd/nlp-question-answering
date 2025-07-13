import math
import numpy as np
import jax.numpy as jnp
import jax
from jax import Array
from jax.typing import ArrayLike
from flax import linen as nn


def scaled_dot_product(q: ArrayLike, k: ArrayLike, v: ArrayLike, mask: ArrayLike=None) -> tuple[Array, Array]:
    """
    Compute the scaled dot product attention.

    Args:
        q: Queries embeddings, in the shape (batch_size, number_of_heads, sequence_length, embedding_dimensionality).
        k: Keys embeddings, in the shape (batch_size, number_of_heads, sequence_length, embedding_dimensionality).
        v: Values embeddings, in the shape (batch_size, number_of_heads, sequence_length, embedding_dimensionality).
        mask: Mask (optional), in the shape (batch_size, number_of_heads, sequence_length, sequence_length).

    Returns:
        Weighted sum of values, in the shape (batch_size, number_of_heads, sequence_length, embedding_dimensionality).
        Attention weights, in the shape (batch_size, number_of_heads, sequence_length, sequence_length).
    """

    sequence_length_q = q.shape[-2]
    sequence_length_k = k.shape[-2]
    sequence_length_v = v.shape[-2]
    assert sequence_length_q==sequence_length_v, \
        "Queries and values must have the same sequence length."
    assert sequence_length_k==sequence_length_v, \
        "Keys and values must have the same sequence length."

    hidden_dimensionality_q = q.shape[-1]
    hidden_dimensionality_k = k.shape[-1]
    assert hidden_dimensionality_q==hidden_dimensionality_k, \
        "Queries and keys must have the same hidden dimensionality."

    # Scaled attention logits
    matmul_qk = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    scaled_attention_logits = matmul_qk / math.sqrt(hidden_dimensionality_k)

    # Apply mask
    if mask is not None:
        scaled_attention_logits = jnp.where(mask==1, -1e9, scaled_attention_logits)

    # Softmax
    attention_weights = nn.softmax(scaled_attention_logits, axis=-1)

    # Weighted sum of values
    weighted_sum_of_values = jnp.matmul(attention_weights, v)

    return weighted_sum_of_values, attention_weights


def expand_mask(mask):
    """
    Expand the mask to four dimensions.

    Args:
        mask: Mask, in two, three, or four dimensions. The two last dimensions are (sequence_length, sequence_length).
        If 3D, the first dimension is (batch_size). If 4D, the two first dimensions are (batch_size, number_of_heads).

    Returns:
        Mask, in the shape (batch_size, number_of_heads, sequence_length, sequence_length).
    """

    assert mask.ndim >= 2, \
        "Mask must have at least two dimensions (sequence_length, sequence_length)."
    assert mask.ndim <= 4, \
        "Mask must have at most four dimensions (batch_size, number_of_heads, sequence_length, sequence_length)."

    # Broadcast over (number_of_heads)
    if mask.ndim == 3:
        mask = jnp.expand_dims(mask, axis=1)

    # Broadcast over (batch_size), then (number_of_heads)
    while mask.ndim < 4:
        mask = jnp.expand_dims(mask, axis=0)

    return mask


class MultiheadAttention(nn.Module):
    embedding_dimensionality : int
    number_of_heads : int

    def setup(self):

        self.qkv_projection = nn.Dense(
            features=3 * self.embedding_dimensionality,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=True,
            bias_init=nn.initializers.zeros)

        self.w_o_projection = nn.Dense(
            features=self.embedding_dimensionality,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=True,
            bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None):

        batch_size, sequence_length, embedding_dimensionality = x.shape

        if mask is not None:
            mask = expand_mask(mask)

        qkv = self.qkv_projection(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.number_of_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)
        q, k, v = jnp.array_split(qkv, indices_or_sections=3, axis=-1)

        weighted_sum_of_values, attention_weights = scaled_dot_product(q, k, v, mask=mask)
        weighted_sum_of_values = weighted_sum_of_values.transpose(0, 2, 1, 3)
        weighted_sum_of_values = weighted_sum_of_values.reshape(batch_size, sequence_length, embedding_dimensionality)

        w_o = self.w_o_projection(weighted_sum_of_values)

        return w_o, attention_weights


class EncoderBlock(nn.Module):
    input_dimensionality : int
    number_of_heads : int
    feedforward_dimensionality : int
    dropout_probability : float

    def setup(self):

        self.self_attention = MultiheadAttention(
            embedding_dimensionality=self.input_dimensionality,
            number_of_heads=self.number_of_heads)

        self.linear = [
            nn.Dense(features=self.feedforward_dimensionality),
            nn.Dropout(self.dropout_probability),
            nn.relu,
            nn.Dense(features=self.input_dimensionality)]

        self.layer_normalization_1 = nn.LayerNorm()
        self.layer_normalization_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_probability)

    def __call__(self, x, mask=None, train=True):

        # Attention
        attention_out, _ = self.self_attention(x, mask=mask)
        x = x + self.dropout(attention_out, deterministic=not train)
        x = self.layer_normalization_1(x)

        # Feedforward
        linear_out = x
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.layer_normalization_2(x)

        return x


class TransformerEncoder(nn.Module):
    number_of_layers : int
    input_dimensionality : int
    number_of_heads : int
    feedforward_dimensionality : int
    dropout_probability : float

    def setup(self):
        self.layers = [
            EncoderBlock(self.input_dimensionality, self.number_of_heads, self.feedforward_dimensionality, self.dropout_probability)\
            for _ in range(self.number_of_layers)]

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            x = l(x, mask=mask, train=train)
        return x

    def get_attention_maps(self, x, mask=None, train=True):
        attention_maps = []
        for l in self.layers:
            _, attention_map = l.self_attention(x, mask=mask)
            attention_maps.append(attention_map)
            x = l(x, mask=mask, train=train)
        return attention_maps


class PositionalEncoding(nn.Module):

    hidden_dimensionality : int
    maximum_sequence_length : int = 5000

    def setup(self):
        positional_encoding = np.zeros((self.maximum_sequence_length, self.hidden_dimensionality))
        position = np.arange(0, self.maximum_sequence_length, dtype=np.float32)[:,None]
        denominator = np.exp(np.arange(0, self.hidden_dimensionality, 2) * (-math.log(10000.0) / self.hidden_dimensionality))
        positional_encoding[:, 0::2] = np.sin(position * denominator)
        positional_encoding[:, 1::2] = np.cos(position * denominator)
        positional_encoding = positional_encoding[None]
        self.positional_encoding = jax.device_put(positional_encoding)

    def __call__(self, x):

        ##print(x.shape)
        ##print(self.positional_encoding[:, :x.shape[1]].shape)
        ##input("pause")

        x = x + self.positional_encoding[:, :x.shape[1]]

        return x
