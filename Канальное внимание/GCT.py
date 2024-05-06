"""Модуль реализующий слой закрытой трансформации по каналам (GCT)"""

import tensorflow as tf
import keras

class GCT(keras.Model):
    """
        Cлой закрытой трансформации по каналам

        Входные данные:
            num_channels - число каналов (фильтров)
            epsilon - константа, предотвращающая деление на 0
            mode - режим нормализации ('l1' или 'l2')
            after_relu - булевая переменная, показывающая применяется ли ReLU

    """
    def __init__(self, num_channels: int, epsilon: float = 1e-5,
                  mode: str = 'l2', after_relu: bool = False):
        super().__init__()

        self.num_channels = num_channels

        self.alpha = self.add_weight(shape=(1,1,1,self.num_channels), 
                                    initializer='ones',  trainable=True)
        self.gamma = self.add_weight(shape=(1,1,1,self.num_channels),
                                    initializer='zeros', trainable=True)
        self.beta = self.add_weight(shape=(1,1,1,self.num_channels),
                                    initializer='zeros', trainable=True)
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def call(self, x):
        """
            Вызов слоя закрытой трансформации по каналам

            Входные данные:
                x - входной тензор
            Выходные данные:
                tf.tensor - результат применения слоя закрытой трансформации по каналам к тензору
        """
        if self.mode == 'l2':
            embedding = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(x, 2), 2, keepdims=True),
                                              3, keepdims=True) + self.epsilon) * self.alpha
            norm = self.gamma / tf.sqrt(tf.reduce_mean(tf.pow(embedding, 2),
                                                        axis=1,keepdims=True) + self.epsilon)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = tf.abs(x)
            else:
                _x = x

            embedding = tf.reduce_sum(tf.reduce_sum(_x, 2, keepdims=True),
                                    3, keepdims=True) * self.alpha
            norm = self.gamma / (tf.reduce_mean(tf.abs(embedding),
                                                axis=1, keepdims=True) + self.epsilon)

        else:
            print("Неизвестный mode")
            return None

        gate = 1. + tf.tanh(embedding*norm + self.beta)

        return x * gate

    def get_config(self):
        """Метод для сохранения модели с использованием данного слоя"""
        return{
            'num_channels' : self.num_channels,
            'epsilon' : self.epsilon,
            'mode' :  self.mode,
            'after_relu' : self.after_relu
        }
