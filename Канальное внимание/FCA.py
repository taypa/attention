"""Модуль реализующий слой внимания по частотным каналам (FCA)"""

import math
import tensorflow as tf
import keras
import numpy as np
from AdaptiveAvgPool2D import AdaptiveAvgPool2D

def get_freq_indices(method: str) -> list:
    """
        Получение списка координат для DCT фильтра

        Входные данные:
            method  - строка, определяющая метод получения координат

        Выходные данные:
            mapper_x - список координат по ширине
            mapper_y - список координат по высоте
    """
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                    'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                    'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])

    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0,
                            0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6,
                            3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0,
                            1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4,
                            3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4,
                            6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2,
                            2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

@keras.saving.register_keras_serializable(package="ECA")
class FCA(keras.Model):
    """
        Слой внимания по частотным каналам

        Входные данные:
            channel - число каналов
            dct_h - высота окна, к которому применяется фильтр
            dct_w - ширина окна, к которому применяется фильтр
            reduction - число, показывающее во сколько раз сжимается исходный тензор
            freq_sel_method - метод выбора частот
    """
    def  __init__(self, channel: int,
                  dct_h: int, dct_w: int,
                  reduction: int = 16, freq_sel_method: str = 'top16'):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        map_x,  map_y = get_freq_indices(freq_sel_method)
        self.num_split = len(map_x)

        mapper_x = [tmp_x * (dct_h // 7) for tmp_x in map_x]
        mapper_y = [tmp_y * (dct_w // 7) for tmp_y in map_y]
        self.dct_layer = DCT(dct_h, dct_w, mapper_x, mapper_y, channel)

        self.fc  = keras.Sequential([
            keras.layers.Dense(channel // reduction, use_bias=False, activation='relu'),
            keras.layers.Dense(channel, use_bias=False, activation='sigmoid')
        ])

        self.avg_pool =  AdaptiveAvgPool2D((self.dct_h, self.dct_w), input_ordering='NHWC')

    def call(self, x):
        """
            Вызов слоя внимания по частотным каналам

            Входные данные:
                x - входной тензор
            Выходные данные:
                tf.tensor - результат применения слоя внимания по частотным каналам к тензору
        """
        b, h, w, c = x.shape

        x_pooled =  x

        if h!= self.dct_h or w!= self.dct_w:
            x_pooled = self.avg_pool(x)

        y = self.dct_layer(x_pooled)
        y = self.fc(y)

        if b is None:
            y = tf.reshape(y,   (-1,1,1, c))
        else:
            y = tf.reshape(y,   (b,1,1, c))

        return x * y
    
    def get_config(self):
        return {
        "channel": self.channel,
        "dct_h": self.dct_h,
        "dct_w": self.dct_w,
        "reduction": self.reduction,
        "freq_sel_method": self.freq_sel_method
    }

class DCT(keras.Model):
    """
        Класс дискретного косинусного преобразования

        Входные данные:
            h - высота окна
            w - ширина окна
            mapper_x - список координат по X,  которые участвуют в фильтре
            mapper_y - список координат по Y,  которые участвуют в фильтре
            channel - число каналов
    """

    def __init__(self, h: int, w: int, mapper_x: list, mapper_y: list, channel: int):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)
        self.weight = tf.cast(self.get_gct_filter(h, w, mapper_x, mapper_y, channel),
                               dtype=tf.float32)

    def call(self, x):
        """
            Вызов DCT фильтра

            x - входной  тензор
        """
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + \
                str(len(x.shape))

        x = tf.cast(x, dtype=tf.float32)
        x = tf.math.multiply(x, self.weight)
        result = tf.reduce_sum(tf.reduce_sum(x, 1), 1)
        return result

    def build_filter(self, pos: int, freq: int, pos_cnt: int) -> float:
        """
            Построение фильтра

            pos - позиция в пространстве
            freq - частота
            pos_cnt - количество позиций
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / pos_cnt) / math.sqrt(pos_cnt)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)


    def get_gct_filter(self, tile_x: int, tile_y: int, mapper_x: list, mapper_y: list, channel: int)  -> np.array:
        """
        Построение фильтра для DCT
        Преобразует пространственные частоты в частоты косинусов

        tile_x - ширина части изоображения,  для которого создается фильтр
        tile_y - высота части изоображения,  для которого создается фильтр
        mapper_x - список координат по X,  которые участвуют в фильтре
        mapper_y - список координат по Y,  которые участвуют в фильтре
        channel - количество каналов
        """
        dct_filter = np.zeros((tile_x, tile_y, channel))
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_x):
                for t_y in range(tile_y):
                    dct_filter[t_x, t_y, i * c_part : (i+1) * c_part] = (
                        self.build_filter(t_x, u_x, tile_x) * self.build_filter(t_y, v_y, tile_y))

        dct_filter = tf.convert_to_tensor(dct_filter)
        return dct_filter
