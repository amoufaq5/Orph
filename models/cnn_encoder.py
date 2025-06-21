# cnn_encoder.py
import tensorflow as tf
from tensorflow.keras import layers, models

class CNNMedicalEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=1024):
        super(CNNMedicalEncoder, self).__init__()

        # ResNet-style CNN backbone
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')

        # Block 1
        self.block1 = tf.keras.Sequential([
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
        ])

        # Block 2
        self.block2 = tf.keras.Sequential([
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
        ])

        # Flatten to embedding vector
        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(embedding_dim, activation='linear')

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_pool(x)
        return self.fc(x)  # returns a [batch_size, embedding_dim] tensor
