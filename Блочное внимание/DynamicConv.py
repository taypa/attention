import tensorflow as tf
import keras
from AdaptiveAvgPool2D import AdaptiveAvgPool2D

class Attention2D(keras.layers.Layer):
    def __init__(self, in_planes, ratios, K, temperature):
        super().__init__()
        # Для уменьшения температуры τ от 30 до 1 линейно в первых 10 эпохах.
        assert temperature % 3 == 1
        self.avgpool = AdaptiveAvgPool2D(1)
        self.in_planes = in_planes
        self.ratios = ratios
        self.K = K
        if in_planes != 3:
            hidden_planes = int(self.in_planes * self.ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = keras.layers.Conv2D(hidden_planes, 1, use_bias=False)
        self.fc2 = keras.layers.Conv2D(self.K, 1, use_bias=True)
        self.temp = temperature

    def update_temperature(self):
        if self.temp != 1:
            self.temp -= 3

    def call(self, z):
        z = self.avgpool(z)
        z = self.fc1(z)
        z = tf.nn.relu(z)
        z = self.fc2(z)
        z = tf.reshape(z, (z.shape[0], -1))

        return keras.layers.Softmax(axis=1)(z / self.temperature)
    
class DynamicConv2D(keras.layers.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34):
        super(DynamicConv2D, self).__init__()

        if in_planes % groups != 0:
            raise ValueError('Ошибка: in_planes%groups != 0')
        self.in_plaanes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = Attention2D(in_planes, ratio, K, temperature)
        self.weight = tf.random.normal((K, out_planes, kernel_size, kernel_size,in_planes // groups))

        if bias:
            self.bias = tf.random.normal((K, out_planes))
        else:
            self.bias = None

    def update_temperature(self):
        self.attention.update_temperature()

    def call(self, z):
        softmax_attention = self.attention(z)
        batch_size, height, width, in_planes = z.shape
        z = tf.reshape(z, (1, height, width, -1))
        weight = tf.reshape(self.weight, (self.K, -1))

        aggregate_weight = tf.matmul(softmax_attention, weight)
        aggregate_weight = tf.reshape(aggregate_weight, (-1, self.in_planes, self.kernel_size, self.kernel_size))

        if self.bias is not None:
            aggregate_bias = tf.matmul(softmax_attention, self.bias)
            aggregate_bias = tf.reshape(aggregate_bias, (-1,))
            output = keras.layers.Conv2D(self.out_planes, self.kernel_size, strides=self.stride, padding=self.padding,
                            dilation_rate=self.dilation, groups=self.groups * batch_size, use_bias=True)(z, aggregate_weight, aggregate_bias)
        else:
            output = self.layers.Conv2D(self.out_planes, self.kernel_size, strides=self.stride, padding=self.padding,
                            dilation_rate=self.dilation, groups=self.groups * batch_size, use_bias=False)(z, aggregate_weight)

        output = tf.reshape(output, (batch_size, self.out_planes, output.shape[-2], output.shape[-1]))
        return output