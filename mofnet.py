import numpy as np

import tensorflow as tf
import tensorflow.keras as keras


class Embedding(keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.w = self.add_weight(shape=[input_dim, output_dim],
                                 trainable=True, name="w")

    def call(self, x):
        indices = tf.where(x < 0, tf.zeros_like(x), x)
        indices = tf.expand_dims(indices, axis=-1)

        y = tf.gather_nd(self.w+0, indices)

        mask = tf.where(x < 0, tf.zeros_like(x), tf.ones_like(x))
        mask = tf.cast(tf.expand_dims(mask, axis=-1), tf.float32)

        return y * mask


class NeighborLookup(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, neighbor_list):
        # Alias.
        n = neighbor_list

        mask = tf.where(n < 0, tf.zeros_like(n), tf.ones_like(n))
        mask = tf.cast(tf.expand_dims(mask, axis=-1), tf.float32)

        batch_indices = tf.reshape(tf.range(n.shape[0]), [-1, 1, 1])
        batch_indices = tf.broadcast_to(batch_indices, n.shape)

        n = tf.where(n < 0, tf.zeros_like(n), n)
        indices = tf.stack([batch_indices, n], axis=-1)

        y = tf.gather_nd(x, indices)

        return y * mask


class GraphConvolution(keras.layers.Layer):
    def __init__(self, node_dim, edge_dim):
        super().__init__()

        z_dim = 2*node_dim + edge_dim

        self.wf = self.add_weight(shape=[z_dim, node_dim],
                                  trainable=True, name="wf")
        self.bf = self.add_weight(shape=[node_dim],
                                  trainable=True, name="bf")
        self.ws = self.add_weight(shape=[z_dim, node_dim],
                                  trainable=True, name="ws")
        self.bs = self.add_weight(shape=[node_dim],
                                  trainable=True, name="bs")

        self.neighbor_lookup = NeighborLookup()

    def call(self, v, nl, e):
        nn = self.neighbor_lookup(v, nl)

        v_view = tf.broadcast_to(tf.expand_dims(v, axis=2), shape=nn.shape)
        z = tf.concat([v_view, nn, e], axis=-1)

        mask = tf.where(nl < 0, tf.zeros_like(nl), tf.ones_like(nl))
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)

        s = tf.sigmoid(tf.einsum("ijkl,lm->ijkm", z, self.wf) + self.bf) \
          * tf.math.tanh(tf.einsum("ijkl,lm->ijkm", z, self.ws) + self.bs) \
          * mask

        v_next = v + tf.reduce_sum(s, axis=2)

        return v_next


class GraphPooling(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, v, n):
        mask = tf.where(n < 0, tf.zeros_like(n), tf.ones_like(n))
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, axis=-1)

        y = tf.reduce_sum(v*mask, axis=1) / tf.reduce_sum(mask, axis=1)

        return y


class MofNet(keras.Model):
    def __init__(self, node_dim, edge_dim):
        super().__init__()

        self.node_embedding = Embedding(100, node_dim)
        self.edge_embedding = Embedding(100, edge_dim)

        n_convs = 3
        self.convs = [
            GraphConvolution(node_dim, edge_dim) for i in range(n_convs)
        ]

        self.pooling = GraphPooling()

        self.dense = keras.layers.Dense(units=1)

    def call(self, node_types, neighbor_list, edge_types, slot_types):
        n = self.node_embedding(node_types)
        n = n + slot_types
        e = self.edge_embedding(edge_types)

        v = n

        nl = neighbor_list
        for conv in self.convs:
            v = conv(v, nl, e)

        v = self.pooling(v, node_types)

        # Save MOF vector.
        self.v = v

        y = self.dense(v)

        return y
