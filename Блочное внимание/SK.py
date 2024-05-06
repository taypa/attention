import tensorflow as tf
import keras
from  AdaptiveAvgPool2D import AdaptiveAvgPool2D

class SK(keras.Model):
    def __init__(self, features,  M=2, G=32, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features // r), L)
        self.features = features
        self.M = M
        self.G = G
        self.r = r
        self.stride = stride
        self.L = L
        self.convs = []

        for i in range(self.M):
            self.convs.append(keras.Sequential([
                keras.layers.Conv2D(filters=self.features,
                                       kernel_size=3, strides=self.stride,
                                       padding="same",dilation_rate=i+1,
                                       groups=self.G,  use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU()
            ]))

        self.gap = AdaptiveAvgPool2D((1,1))
        self.fc = keras.Sequential([
            keras.layers.Conv2D(filters=d, kernel_size=1,
                                   strides = 1, use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])
        self.fcs = []
        for i in range(self.M):
            self.fcs.append(
                keras.layers.Conv2D(filters=self.features, kernel_size=1,
                                       strides = 1)
            )
        self.softmax = keras.layers.Softmax(axis=1)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        #batch_size = x.shape[0]
        feats = [conv(x) for conv in self.convs if conv(x) is not None]
        feats = tf.concat(feats, axis=3)
        feats = tf.reshape(feats, (batch_size, feats.shape[1],
                                   feats.shape[2], self.M, self.features,))
        featsU = tf.reduce_sum(feats, axis=1)
        featsS = self.gap(featsU)
        featsZ =   self.fc(featsS)

        attn_vectors = [fc(featsZ) for fc in self.fcs if fc(featsZ) is not None]
        attn_vectors = tf.concat(attn_vectors, axis=3)
        attn_vectors = tf.reshape(attn_vectors,(batch_size,  1, 1,  self.M, self.features,))
        attn_vectors = self.softmax(attn_vectors)

        featsV = tf.reduce_sum(feats*attn_vectors,axis=3)

        return featsV
    
    def get_config(self):
        return {
            "features": self.features,
            "M" : self.M,
            "G" : self.G,
            "r" : self.r,
            "stride" :self.stride,
            "L" : self.L
        }