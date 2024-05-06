"""Модуль реализующий слой эффективного внимания (ECA)"""

import keras
@keras.saving.register_keras_serializable(package="ECA")
class ECA(keras.Model):
    """
        Слой эффективного внимания

        Входные данные:
            kernel_size - размерность ядра свертки
    """
    def __init__(self, kernel_size: int=3):
        super().__init__()
        self.avg_pool = keras.layers.AveragePooling2D(1)
        self.conv = keras.layers.Conv2D(1, kernel_size=kernel_size, 
                                        padding="same", activation='sigmoid', use_bias=False)

    def call(self, x):
        """
            Вызов слоя эффективного внимания

            Входные данные:
                x - входной тенызор
            Выходные данные:
                tf.tensor - результат применения слоя эффективного внимания к тензору
        """
        y = self.avg_pool(x)
        y = self.conv(y)

        return x * y
    
    def get_config(self):
        return {
        "k_size": self.k_size
        }