import tensorflow as tf
from tensorflow.keras import layers, models

class OrphTransformer(tf.keras.Model):
    def __init__(
        self,
        vocab_size=32000,
        max_len=512,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
        ff_dim=4096,
        dropout_rate=0.1,
        use_segment_embedding=True,
        use_modality_embedding=True,
    ):
        super(OrphTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        # Embedding layers
        self.token_embedding = layers.Embedding(vocab_size, hidden_size)
        self.position_embedding = layers.Embedding(max_len, hidden_size)
        self.segment_embedding = layers.Embedding(2, hidden_size) if use_segment_embedding else None
        self.modality_embedding = layers.Embedding(3, hidden_size) if use_modality_embedding else None

        # Transformer encoder blocks
        self.encoder_layers = [
            layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)
        ]
        self.attention_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size // num_heads)
            for _ in range(num_layers)
        ]
        self.ffn_layers = [
            tf.keras.Sequential([
                layers.Dense(ff_dim, activation='relu'),
                layers.Dense(hidden_size),
                layers.Dropout(dropout_rate)
            ]) for _ in range(num_layers)
        ]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        input_ids, segment_ids, modality_ids = inputs
        seq_len = tf.shape(input_ids)[1]

        x = self.token_embedding(input_ids)
        x += self.position_embedding(tf.range(seq_len))

        if self.segment_embedding:
            x += self.segment_embedding(segment_ids)
        if self.modality_embedding:
            x += self.modality_embedding(modality_ids)

        for i in range(self.num_layers):
            attn_output = self.attention_layers[i](x, x)
            x = self.encoder_layers[i](x + attn_output)
            ffn_output = self.ffn_layers[i](x)
            x = self.encoder_layers[i](x + ffn_output)

        return x
