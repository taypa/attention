"""Модуль реализующий слой сжатия и возбуждения (SE)"""

import keras

class SE(keras.Model):
    """
        Слой сжатия и возббуждения

        Входные данные:
            reduction - число, показывающее во сколько раз сжимается исходный тензор
    """
    def __init__(self,channel, reduction: int = 16):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        self.avg_pool= keras.layers.GlobalAveragePooling2D('channels_last', True)

        self.fc= keras.Sequential([
            keras.layers.Dense(channel // reduction, use_bias=False),
            keras.layers.ReLU(),
            keras.layers.Dense(channel, use_bias=False,  activation='sigmoid')
          ])

    def call(self, x):
        """
            Вызов слоя эффективного внимания

            Входные данные:
                x - входной тенызор
            Выходные данные:
                tf.tensor - результат применения слоя эффективного внимания к тензору
        """
        if x[0] is None:
            x.shape[0] = 1

        y = self.avg_pool(x)
        y = self.fc(y)

        return x * y
    
    def get_config(self):
        """Метод для сохранения модели с использованием данного слоя"""
        return{
            'channel' : self.channel,
            'reduction' : self.reduction
        }