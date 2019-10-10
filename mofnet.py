import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

# B: batch size.
# N: the maximum number of atoms in the batch.
# L: the maximum size of neighbor list in the batch.
# E: embedding dimension size.

class Embedding(keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        """
        Arguments:
            input_dim: the number of categories.
            output_dim: the size of embedding.
        """
        super().__init__()

        self.w = self.add_weight(shape=[input_dim, output_dim],
                                 trainable=True, name="w")

        self.dropout = keras.layers.Dropout(0.5)

    def call(self, x, training=False):
        """
        Arguments:
            x: tensor of size (B, N).

        Returns:
            y: embedded tensor of size (B, N, E).
        """
        indices = tf.where(x < 0, tf.zeros_like(x), x)
        # indices size = (B, N, 1).
        indices = tf.expand_dims(indices, axis=-1)

        if training:
            w = self.dropout(self.w)
        else:
            w = self.w + 0

        # y size = (B, N, E).
        y = tf.gather_nd(w, indices)

        # Make dummy embedding to zero vector.
        mask = tf.where(x < 0, tf.zeros_like(x), tf.ones_like(x))
        # mask size = (B, N, 1).
        mask = tf.cast(tf.expand_dims(mask, axis=-1), tf.float32)

        return y * mask


class NeighborLookup(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, neighbor_list):
        """
        Arguments:
            x: tensor of size (B, N, X).
            neighbor_list: integer tensor of size (B, N, L).

        Returns:
            y: tensor of size (B, N, L, X).
        """
        # Alias.
        n = neighbor_list

        mask = tf.where(n < 0, tf.zeros_like(n), tf.ones_like(n))
        # mask size = (B, N, L, 1).
        mask = tf.cast(tf.expand_dims(mask, axis=-1), tf.float32)

        # batch_indices size = (B, 1, 1).
        batch_indices = tf.reshape(tf.range(n.shape[0]), [-1, 1, 1])
        # batch_indices size = (B, N, L).
        batch_indices = tf.broadcast_to(batch_indices, n.shape)

        n = tf.where(n < 0, tf.zeros_like(n), n)
        # indices size = (B, N, L, 2)
        # The last dimension contains (batch_index, neighbor_index).
        indices = tf.stack([batch_indices, n], axis=-1)

        # Batch-wise neighbor selections are conducted.
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
        self.batchnorm = keras.layers.BatchNormalization()

    def apply_batch_normalization(self, v, training=False):
        B, N, X = v.shape

        # v_flat size = (B*N, X).
        v_flat = tf.reshape(v, shape=[-1, X])

        # True for valid row index and False for row index of padded vectors.
        idx = tf.reduce_all(tf.abs(v_flat) > 1e-5, axis=-1)

        # Gather valid node vectors from the tensor.
        v_valid = v_flat[idx]

        # Apply batch normalization to the node vectors.
        v_batch = self.batchnorm(v_valid, training=training)

        # Return to original shape.

        # Get the values of batch normalized v as 1D array.
        v_batch_values = tf.reshape(v_batch, [-1])

        # v_batch_indices shape = (B*N, X).
        # All elements are the same for each row.
        # E.g. v_batch_indices =
        # [[T, T, T, ..., T],
        #  [F, F, F, ..., F],
        #  ...,
        #  [T, T, T, ..., T]]
        v_batch_indices = tf.broadcast_to(
            tf.reshape(idx, shape=[-1, 1]),
            v_flat.shape,
        )

        # Get the index of the True positions.
        v_batch_indices = tf.where(v_batch_indices)

        # Make spare tensor.
        sparse_tensor = tf.sparse.SparseTensor(
                            indices=v_batch_indices,
                            values=v_batch_values,
                            dense_shape=v_flat.shape
                        )

        # Return to dense tensor.
        dense_tensor = tf.sparse.to_dense(sparse_tensor)

        # Return to origianl shape, (B, N, X).
        v_after_batchnorm = tf.reshape(dense_tensor, v.shape)

        return v_after_batchnorm

    def call(self, v, nl, e, training=False):
        """
        Arguments:
            v: input tensor (node vectors) of size (B, N, X).
            nl: neighbot list of size (B, N, L).
            e: edge embedding tensor of size (B, N, L, Y).

        Returns:
            v_next: next node vectors of size (B, X').
        """
        # nn size = (B, N, L, X).
        nn = self.neighbor_lookup(v, nl)

        # v -> v_view = (B, N, X) -> (B, N, L, X).
        v_view = tf.broadcast_to(tf.expand_dims(v, axis=2), shape=nn.shape)
        # Similar to z of the CGCNN. z size = (B, N, L, 2X+Y).
        z = tf.concat([v_view, nn, e], axis=-1)

        mask = tf.where(nl < 0, tf.zeros_like(nl), tf.ones_like(nl))
        mask = tf.cast(mask, tf.float32)
        # mask size = (B, N, L, 1).
        mask = tf.expand_dims(mask, axis=-1)

        # s size = (B, N, L, X')
        s = tf.sigmoid(tf.einsum("ijkl,lm->ijkm", z, self.wf) + self.bf) \
          * tf.math.tanh(tf.einsum("ijkl,lm->ijkm", z, self.ws) + self.bs) \
          * mask

        # Warning: you can use the tf.reduce_sum but can not use tf.reduce_mean
        # because of the mask. If you want to average the summation, you should
        # devide the nodes sum by the sum of the mask.
        v_next = v + tf.reduce_sum(s, axis=2)

        # Apply batch normalization after the activation.
        v_next = self.apply_batch_normalization(v_next, training)

        return v_next


class GraphPooling(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, v, n):
        """
        Arguments:
            v: node vertors of size (B, N, X).
            n: node type tensor of size (B, N).

        Returns:
            y: averaged node tensor of size (B, X). So the batch has unique
                vector.
        """

        mask = tf.where(n < 0, tf.zeros_like(n), tf.ones_like(n))
        mask = tf.cast(mask, tf.float32)
        # mask size = (B, N, 1).
        mask = tf.expand_dims(mask, axis=-1)

        # Manual average needed due to the mask.
        y = tf.reduce_sum(v*mask, axis=1) / tf.reduce_sum(mask, axis=1)

        return y


class MofNet(keras.Model):
    def __init__(self, node_dim, edge_dim, n_convs=3, outdim=1):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim

        self.node_embedding = Embedding(1000, node_dim)
        self.edge_embedding = Embedding(1000, edge_dim)

        self.convs = [
            GraphConvolution(node_dim, edge_dim) for i in range(n_convs)
        ]

        self.pooling = GraphPooling()
        self.dense = keras.layers.Dense(units=outdim)

    def initialize_weights(self):
        m = self.edge_dim

        nt = np.zeros(shape=[1, 1], dtype=np.int32)
        nl = np.zeros(shape=[1, 1, 1], dtype=np.int32)
        et = np.zeros(shape=[1, 1, 1], dtype=np.int32)
        st = np.zeros(shape=[1, 1, m], dtype=np.float32)

        # Initialize weights by calling "__call__".
        self(nt, nl, et, st)

    def call(self, node_types, neighbor_list, edge_types, slot_types,
             training=False):
        n = self.node_embedding(node_types, training)
        n = n + slot_types
        e = self.edge_embedding(edge_types, training)

        v = n

        nl = neighbor_list
        for conv in self.convs:
            v = conv(v, nl, e, training)

        v = self.pooling(v, node_types)

        # Save MOF vector.
        self.v = v

        y = self.dense(v)

        return y
