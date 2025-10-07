import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline



def make_mlp(input_dim: int, n_classes: int) -> tf.keras.Model:
    """FC MLP: [512→256→128], He init, BN+ReLU, Dropout=0.45, softmax output."""
    inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)

    x = tf.keras.layers.Dense(512, activation=None, kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.45)(x)

    x = tf.keras.layers.Dense(256, activation=None, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.45)(x)

    x = tf.keras.layers.Dense(128, activation=None, kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.45)(x)

    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def train_mlp_keras(X_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> tf.keras.Model:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = make_mlp(X_train.shape[1], n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=50, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.7, patience=28, min_lr=1e-5, verbose=1
        ),
    ]

    model.fit(
        X_train_scaled, y_train,
        epochs=1000, batch_size=128, shuffle=True, verbose=1,
        callbacks=callbacks,
    )

    model._train_scaler = scaler
    return model


def train_mlp_sklearn(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            solver="adam",
            learning_rate="constant",
            learning_rate_init=1e-3,
            max_iter=1000,
            random_state=903100,
            n_iter_no_change=20,
            tol=1e-4,
            verbose=False,
            beta_1=0.9, beta_2=0.999, epsilon=1e-8,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe

