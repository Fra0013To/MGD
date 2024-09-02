import tensorflow as tf
from multi_objective_problems.moo_problems import MultiObjectiveProblem

# ---------------- VIENNET MO-FUNCTIONS (START) ----------------------
@tf.function
def __viennet(X):
    Y0 = 0.5 * tf.reduce_sum(X ** 2, axis=1) + tf.sin(tf.reduce_sum(X ** 2, axis=1))
    Y1 = 0.125 * ((3 * X[:, 0] - 2 * X[:, 1] + 4) ** 2) + ((X[:, 0] - X[:, 1] + 1) ** 2) / 27 + 15
    Y2 = 1 / (tf.reduce_sum(X ** 2, axis=1) + 1) - 1.1 * tf.exp(-tf.reduce_sum(X ** 2, axis=1))
    Y = tf.concat(
        [
            tf.expand_dims(Y0, axis=-1),
            tf.expand_dims(Y1, axis=-1),
            tf.expand_dims(Y2, axis=-1),
        ],
        axis=-1
    )
    return Y


viennet = MultiObjectiveProblem(function=__viennet)

# ---------------- VIENNET MO-FUNCTIONS (END) ----------------------


# ---------------- KURSAWE MO-FUNCTIONS (START) ----------------------
@tf.function
def __kursawe(X):
    Y0 = -10 * tf.exp(-0.2 * tf.sqrt(tf.reduce_sum(X[:, :2] ** 2, axis=1)))
    Y0 += -10 * tf.exp(-0.2 * tf.sqrt(tf.reduce_sum(X[:, 1:] ** 2, axis=1)))
    Y1 = tf.reduce_sum(tf.pow(tf.abs(X), 0.8) + 5 * tf.sin(X ** 3), axis=1)
    Y = tf.concat(
        [
            tf.expand_dims(Y0, axis=-1),
            tf.expand_dims(Y1, axis=-1),
        ],
        axis=-1
    )
    return Y


kursawe = MultiObjectiveProblem(function=__kursawe)

# ---------------- KURSAWE MO-FUNCTIONS (END) ----------------------


# ---------------- FONSECA-FLEMING MO-FUNCTIONS (START) ----------------------
@tf.function
def __fonsecafleming(X):
    n = tf.cast(X.shape[1], dtype=X.dtype)
    Y0 = 1. - tf.math.exp(- tf.reduce_sum((X - (1. / tf.math.sqrt(n))) ** 2, axis=1))
    Y1 = 1. - tf.math.exp(- tf.reduce_sum((X + (1. / tf.math.sqrt(n))) ** 2, axis=1))
    Y = tf.concat(
        [
            tf.expand_dims(Y0, axis=-1),
            tf.expand_dims(Y1, axis=-1),
        ],
        axis=-1
    )
    return Y


fonsecafleming = MultiObjectiveProblem(function=__fonsecafleming)

# ---------------- FONSECA-FLEMING MO-FUNCTIONS (END) ----------------------

