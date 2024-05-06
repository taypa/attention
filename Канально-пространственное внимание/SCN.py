"""Модуль реализующий слой самокалибрующейся свертки(SCN)"""

import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="SCN")
class SCN(keras.Model):
    """
        Слой самокалибрующейся свертки

        Входные данные:
            filters - количество фильтров для сверток
            stride - шаг для свертки
            padding - тип заполнения после операции свертки
            dilation - коэффициент дилатации для управления расстоянием между точками ядра
            groups - коэффициент, управляющий соединениями между входными данными и выходными
            pooling_rate - размер окна для пулинга
            norm_layer - слой для нормализации 
    """
    def __init__(self, filters: int, stride: int, padding: str,
                 dilation: int, groups: int, pooling_r: int, norm_layer):
        super().__init__()
        self.filters = filters
        self.stride = stride
        self.padding = padding
        self.dilation= dilation
        self.groups = groups
        self.pooling_r = pooling_r
        self.norm_layer  = norm_layer
        self.k2 = keras.Sequential([
            keras.layers.AvgPool2D(pool_size=pooling_r, strides=pooling_r),
            keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1,
                                padding=padding, dilation_rate=dilation,
                                groups=groups, use_bias=False),
            norm_layer()])
        self.k3 = keras.Sequential([
             keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1,
                                padding=padding, dilation_rate=dilation,
                                groups=groups, use_bias=False),
            norm_layer()])
        self.k4 = keras.Sequential([
             keras.layers.Conv2D(filters=filters, kernel_size=3, strides=stride,
                                padding=padding, dilation_rate=dilation,
                                groups=groups, use_bias=False),
            norm_layer()])
    def call(self, x):
        """
            Вызов слоя самокалибрующейся свертки

            Входные данные:
                x - входной тензор
            Выходные данные:
                tf.tensor - результат применения слоя самокалибрующейся свертки к тензору
        """
        identity = x
        tmp = tf.image.resize(self.k2(x), size=identity.shape[1:3],
                               method=tf.image.ResizeMethod.BILINEAR)
        out = tf.sigmoid(identity + tmp)
        out = self.k3(x) * out
        out = self.k4(out)
        return out
    
    def get_config(self):
        """Метод для сохранения модели с использованием данного слоя"""
        return {
            'filters': self.filters,
            'stride': self.stride,
            'padding':  self.padding,
            'dilation' : self.dilation,
            'groups': self.groups,
            'pooling_r': self.pooling_r,
            'norm_layer': self.norm_layer
        }
    