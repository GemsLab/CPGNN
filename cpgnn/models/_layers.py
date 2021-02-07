from . import tf, np
import math
import re
import warnings


class SparseDropout(tf.keras.layers.Layer):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    @tf.function
    def call(self, input: tf.sparse.SparseTensor):
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(input.values.shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(input, dropout_mask)
        return pre_out / keep_prob


class SparseDense(tf.keras.layers.Layer):
    def __init__(self, output_dim, use_bias=False, activation=None,
                 kernel_regularizer=None):
        super().__init__()
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[int(input_shape[-1]), self.output_dim],
            regularizer=self.kernel_regularizer
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[self.output_dim],
                initializer=tf.zeros_initializer
            )
        super().build(input_shape)

    @tf.function
    def call(self, input: tf.sparse.SparseTensor):
        input = tf.sparse.sparse_dense_matmul(input, self.kernel)
        if self.use_bias:
            input += self.bias
        if self.activation:
            input = self.activation(input)
        return input


class GCNLayer(tf.keras.layers.Layer):
    SIGNATURE = ["adjhops", "inputs"]

    def __init__(self, hops=None):
        super().__init__()
        self.hops = hops
        self.cpu_large_spmatmul = False

    @tf.function
    def sparse_dense_matmul(self, sp_a, b, ind: str):
        # ind argument is to force tensorflow to retrace for different hops
        if (sp_a.values.shape[0] * b.shape[1] > (2 ** 31) and
                len(tf.config.experimental.list_physical_devices('GPU')) > 0):
            numSplits = (sp_a.values.shape[0] * b.shape[1] // 2 ** 31) + 1
            splitSizes = np.arange(
                b.shape[1] + numSplits - 1, b.shape[1] - 1, -1) // numSplits
            print(
                f"Splitting tensor to {splitSizes} allow sparse tensor multiplication...")
            assert sum(splitSizes) == b.shape[1]
            b_splits = tf.split(b, splitSizes, axis=-1)
            return tf.concat([tf.sparse.sparse_dense_matmul(sp_a, x) for x in b_splits], axis=-1)
        else:
            return tf.sparse.sparse_dense_matmul(sp_a, b)

    @tf.function
    def call(self, adjhops, inputs):
        return tf.stack([self.sparse_dense_matmul(x, inputs, str(ind)) for ind, x in enumerate(adjhops)
                         if (self.hops is None or ind in self.hops)], axis=-2)


class TensorDotLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, use_bias=False, activation=None,
                 kernel_regularizer=None, sparse_input=False):
        super().__init__()
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer
        self.sparse_input = sparse_input

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[int(input_shape[-2]),
                   int(input_shape[-1]), self.output_dim],
            regularizer=self.kernel_regularizer
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=[int(input_shape[-2]), self.output_dim],
                initializer=tf.zeros_initializer
            )
        super().build(input_shape)

    @tf.function
    def call(self, inputs: tf.sparse.SparseTensor):
        if self.sparse_input:
            raise NotImplementedError()
            # TODO: test the following block
            if len(inputs.shape) == 2:
                inputs = tf.transpose(
                    tf.sparse.sparse_dense_matmul(inputs, self.kernel),
                    perm=[1, 0, 2])
        else:
            inputs = tf.squeeze(tf.matmul(tf.expand_dims(
                inputs, -2), tf.expand_dims(self.kernel, 0)))
            if self.use_bias:
                inputs += tf.expand_dims(self.bias, 0)
        if self.activation:
            inputs = self.activation(inputs)
        return inputs


class GCNTensorDotLayer(tf.keras.layers.Layer):
    ARGS = ["adjhops", "inputs"]

    def __init__(self, output_dim, use_bias=False, activation=None):
        super().__init__()
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.model_regularizer = None
        self.sparse_input = False
        self.kernel = None

    @tf.function
    def call(self, inputs: tf.sparse.SparseTensor, adjhops):
        if self.kernel is None:
            self.kernel = [self.add_weight(
                name="kernel",
                shape=[int(inputs.shape[-1]), self.output_dim],
                regularizer=self.model_regularizer
            ) for _ in range(len(adjhops))]
            if self.use_bias:
                self.bias = self.add_weight(
                    name="bias",
                    shape=[len(adjhops), self.output_dim],
                    initializer=tf.zeros_initializer
                )
        if self.sparse_input:
            inputs = tf.stack([tf.sparse.sparse_dense_matmul(inputs, x)
                               for ind, x in enumerate(self.kernel)])
        else:
            inputs = tf.stack([tf.matmul(inputs, x)
                               for ind, x in enumerate(self.kernel)])
        inputs = tf.stack([tf.sparse.sparse_dense_matmul(adj, inputs[ind])
                           for ind, adj in enumerate(adjhops)], axis=-2)
        if self.use_bias:
            inputs += tf.expand_dims(self.bias, 0)
        if self.activation:
            inputs = self.activation(inputs)
        return inputs


class ConcatLayer(tf.keras.layers.Layer):
    def __init__(self, tags, axis=-1, addInputs=True):
        super().__init__()
        self.tags = tags
        self.axis = axis
        self.addInputs = addInputs

    def call(self, *args, **kwargs):
        selected = [value for name, value in kwargs.items()
                    if name in self.tags]
        if self.addInputs:
            return tf.concat(list(args) + selected, self.axis)
        else:
            return tf.concat(selected, self.axis)


class SumLayer(tf.keras.layers.Layer):
    def __init__(self, dim=-2):
        super().__init__()
        self.dim = dim

    def call(self, inputs):
        return tf.reduce_sum(inputs, self.dim)


class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, loadTag, sliceObj, **kwargs):
        super().__init__()
        self.tag = loadTag
        self.sliceObj = sliceObj

    def call(self, inputs, **kwargs):
        if self.tag:
            inputs = kwargs[self.tag]
        return inputs[:, self.sliceObj]


class CompatibilityLayer(tf.keras.layers.Layer):
    @staticmethod
    def makeDoubleStochasticH(H, max_iterations=float('inf'), delta=1e-7):
        converge = False
        prev_H = H
        i = 0
        while not converge and i < max_iterations:
            prev_H = H
            H /= tf.reduce_sum(H, axis=0, keepdims=True)
            H /= tf.reduce_sum(H, axis=1, keepdims=True)

            delta = tf.linalg.norm(H - prev_H, ord=1)
            if delta < 1e-12:
                converge = True
            i += 1
        if i == max_iterations:
            warnings.warn(
                "makeDoubleStochasticH: maximum number of iterations reached.")
        return H

    @staticmethod
    def makeSymmetricH(H):
        return 0.5 * (H + tf.transpose(H))

    @classmethod
    def estimateH(cls, adj, y, inputs=None, sample_mask=None):
        RWNormAdj = (adj / tf.sparse.reduce_sum(adj, axis=1, keepdims=True))
        if inputs is not None and sample_mask is not None:
            y = tf.cast(y, inputs.dtype)
            sample_mask = tf.cast(sample_mask, inputs.dtype)
            inputs = tf.nn.softmax(inputs)
            inputs = inputs * \
                (1 - sample_mask[:, None]) + y * sample_mask[:, None]
            y = y * sample_mask[:, None]
        else:
            inputs = tf.cast(y, RWNormAdj.dtype)
        nodeH = tf.sparse.sparse_dense_matmul(RWNormAdj, inputs)
        H = tf.concat([
            tf.reduce_mean(tf.gather(nodeH, tf.where(y[:, i]), axis=0), axis=0) for i in range(y.shape[1])
        ], axis=0)
        H_nan = tf.math.is_nan(H)
        if tf.reduce_any(H_nan):
            H = tf.where(H_nan, tf.transpose(H), H)
            H_nan = tf.math.is_nan(H)

        if tf.reduce_any(H_nan):
            H = tf.where(H_nan, 0, H)
            H_miss = (1 - tf.reduce_sum(H, axis=1, keepdims=True))
            H_miss /= tf.reduce_sum(tf.cast(H_nan, H.dtype),
                                    axis=1, keepdims=True)
            H = tf.where(H_nan, H_miss, H)
        H = cls.makeDoubleStochasticH(H, max_iterations=3000)
        return H


class LinBPLayer(tf.keras.layers.Layer):
    ARGS = ["adj", "inputs", "y_train", "train_mask", "adjhops"]

    def __init__(self, iterations=2,
                 glorot_init=False, nonlinear=None,
                 useadjhops=None, nonlinearH=None,
                 zeroregweight=1.0, notrain=False):
        super().__init__()
        self.iterations = iterations
        self.H_hat = None
        self.numLabels = None
        self.glorot_init = glorot_init
        if nonlinear:
            self.non_linear = getattr(tf.nn, nonlinear)
        else:
            self.non_linear = tf.identity
        if nonlinearH:
            self.non_linear_H = getattr(tf.nn, nonlinearH)
        else:
            self.non_linear_H = tf.identity

        self.use_adj_hops = useadjhops
        self.p_weights = 1.0
        self.zero_reg_weight = 1.0
        self.no_train = notrain

    def call(self, inputs, adj, y_train, train_mask, adjhops):
        self.numLabels = y_train.shape[1]
        if self.H_hat is None:
            regularizer = None

            self.H_hat = self.add_weight(
                name="LinBP_H_hat",
                shape=[int(y_train.shape[1]), int(y_train.shape[1])],
                regularizer=regularizer,
                trainable=(not self.no_train)
            )
            if not self.glorot_init:
                H_init = CompatibilityLayer.estimateH(
                    adj, y_train, inputs, train_mask)
                H_init = CompatibilityLayer.makeSymmetricH(H_init)

                H_init -= (1 / y_train.shape[1])
                self.H_hat.assign(H_init)
                self.H_init = H_init

        prior_belief = tf.nn.softmax(inputs)
        E_hat = prior_belief - (1 / y_train.shape[1])
        B_hat = E_hat

        if self.use_adj_hops is not None:
            bp_adj = adjhops[self.use_adj_hops]
        else:
            bp_adj = adj

        for i in range(self.iterations):
            H_hat = self.non_linear_H(self.H_hat)

            B_hat_update = E_hat + \
                self.p_weights * \
                tf.sparse.sparse_dense_matmul(bp_adj, B_hat @ H_hat)

            B_hat_update = self.non_linear(B_hat_update)
            B_hat = B_hat_update

        post_belief = B_hat + (1 / y_train.shape[1])
        self.add_loss(self.zero_reg_weight * tf.linalg.norm(
            tf.reduce_sum(self.non_linear_H(self.H_hat), axis=-1), ord=1))
        return post_belief

    def get_H(self):
        return self.non_linear_H(self.H_hat) + (1 / self.numLabels)


experimentalDict = {
}
