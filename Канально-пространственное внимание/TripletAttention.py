import tensorflow  as tf
import keras

class BaseConv(keras.Model):
    """
        Слой свертки, использующий двумерную свертку, пакетную нормализацию и функцию активации ReLU

        Входные данные:
            out_channels: Количество выходных каналов
            kernel_size: Размер ядра свертки
            stride: Шаг свертки 
            padding: Тип заполнения для свертки
            dilation: Скорость дилатации для свертки
            groups: Количество групп для групповой свертки
            relu: Применять ли активацию ReLU 
            bn: Применять ли пакетную нормализацию
            bias: Включать ли смещение в свертку
    """
    def  __init__(self, out_channels, kernel_size,
                  stride=1, padding='same', dilation=1,
                  groups=1,relu=True,bn=True,bias=False):
        super().__init__()
        self.out_channels = out_channels
        self.conv = keras.layers.Conv2D(out_channels,
                                        kernel_size, strides=stride,
                                        padding=padding,groups=groups,
                                        dilation_rate=dilation,use_bias=bias)
        self.batchnorm = (
            keras.layers.BatchNormalization(
                axis=-1,
                epsilon=1e-5,
                momentum=0.01,
                scale=True,
            )
            if bn
            else None
        )
        self.relu = keras.layers.ReLU() if relu else None
    def call(self, x):
        """
            Вызов свертки

            Входные данные:
                x: входной тензор
            Выходные данные:
                tf.tensor: результат применения слоя свертки к тензору
        """
        x = self.conv(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(keras.Model):
    """
        Слой вычисляющий максимальное и среднее по каналам
    """
    def __init__(self):
        super().__init__()

    def call(self, x):
        """
            Вызов слоя
        """
        max_x = tf.expand_dims(tf.reduce_max(x, axis=3), axis=3)
        mean_x = tf.expand_dims(tf.reduce_mean(x, axis=3), axis=3)
        return tf.concat([max_x, mean_x], axis=3)
    
class AttentionGate(keras.Model):
    """
        Слой внимания
    """
    def __init__(self):
        super().__init__()
        kernel_size = 3
        self.compress = ZPool()
        self.conv = BaseConv(
            1, kernel_size, relu=False
        )

    def call(self, x):
        """
            Вызов слоя внимания

            Входные данные:
                x: входной тензор
            Выходные данные:
                tf.tensor: результат применения слоя внимания к тензору
        """
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = keras.activations.sigmoid(x_out)
        return x * scale
    
class TripletAttention(keras.Model):
    """
        Слой тройного внимания

        Входные данные:
            no_spatial: булевская переменная, указывающая, вычисляется ли пространственное внимание
    """
    def __init__(self, no_spatial=False):
        super().__init__()
        self.no_spatial = no_spatial
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def call(self, x):
        """
            Вызов слоя тройного внимания

            Входные данные:
                x: входной тензор
            Выходные данные:
                tf.tensor: результат применения слоя тройного внимания к тензору
        """
        x_perm1 = tf.transpose(x, perm=[0, 2, 1, 3])
        x_out1 = self.cw(x_perm1)
        x_out11 = tf.transpose(x_out1, perm=[0, 2, 1, 3])

        x_perm2 = tf.transpose(x, perm=[0, 3, 2, 1])
        x_out2 = self.hc(x_perm2)
        x_out21 = tf.transpose(x_out2, perm=[0, 3, 2, 1])
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out
    
    def get_config(self):
        """Метод для сохранения модели с использованием данного слоя"""
        return {
            'no_spatial': self.no_spatial
        }