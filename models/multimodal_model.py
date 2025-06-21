# multimodal_model.py
import tensorflow as tf
from models.orph_transformer_v2 import OrphTransformer
from models.cnn_encoder import CNNMedicalEncoder

class OrphMultimodal(tf.keras.Model):
    def __init__(self, 
                 vocab_size=32000,
                 max_len=512,
                 hidden_size=1024,
                 num_layers=12,
                 num_heads=16,
                 ff_dim=4096,
                 dropout_rate=0.1):
        super(OrphMultimodal, self).__init__()

        self.transformer = OrphTransformer(
            vocab_size=vocab_size,
            max_len=max_len + 1,  # +1 for image token
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )

        self.image_encoder = CNNMedicalEncoder(embedding_dim=hidden_size)
        self.cls_head = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1, activation='sigmoid')  # binary for now
        ])

    def call(self, inputs, training=False):
        input_ids, segment_ids, modality_ids, image = inputs
        image_vector = self.image_encoder(image)  # [batch, hidden]
        image_vector = tf.expand_dims(image_vector, 1)  # [batch, 1, hidden]

        # Generate token embeddings from transformer
        text_embeddings = self.transformer.token_embedding(input_ids)
        combined = tf.concat([image_vector, text_embeddings], axis=1)

        # Replace position embeddings (shifted by 1)
        seq_len = tf.shape(input_ids)[1] + 1
        position_embeddings = self.transformer.position_embedding(tf.range(seq_len))
        combined += position_embeddings

        # Segment/modality support
        if self.transformer.segment_embedding:
            segment_ids = tf.concat([tf.zeros((tf.shape(segment_ids)[0], 1), dtype=tf.int32), segment_ids], axis=1)
            combined += self.transformer.segment_embedding(segment_ids)

        if self.transformer.modality_embedding:
            modality_ids = tf.concat([tf.ones((tf.shape(modality_ids)[0], 1), dtype=tf.int32), modality_ids], axis=1)
            combined += self.transformer.modality_embedding(modality_ids)

        # Forward through transformer encoder
        for i in range(self.transformer.num_layers):
            attn_output = self.transformer.attention_layers[i](combined, combined)
            combined = self.transformer.encoder_layers[i](combined + attn_output)
            ffn_output = self.transformer.ffn_layers[i](combined)
            combined = self.transformer.encoder_layers[i](combined + ffn_output)

        # Use image+text [CLS] token (first token)
        cls_token = combined[:, 0, :]
        return self.cls_head(cls_token)
