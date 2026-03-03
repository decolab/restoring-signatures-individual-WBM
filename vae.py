"""
Variational Autoencoder (VAE)
=============================

A VAE that maps functional connectivity (FC) matrices into a 2D latent
space for visualisation and comparison of brain states.

Architecture (from the paper):

    Input:   Flattened FC matrix (82×82 = 6724 values)
    Encoder: Dense(1028, relu) → z_mean, z_log_var
    Latent:  2D (via reparameterisation trick)
    Decoder: Dense(1028, relu) → Dense(6724, sigmoid)

    Loss = binary_crossentropy × original_dim + KL divergence

The training set uses three conditions: anesthesia, CT 5V, VL 5V.

Dependencies: numpy, tensorflow (keras)
"""

import numpy as np

# --- Lazy import of TensorFlow/Keras ---
# TensorFlow is only needed for steps 07-08 (VAE training/encoding).
# All other scripts work fine without it.
KERAS_AVAILABLE = False

try:
    import keras
    from keras import layers, ops, Model
    KERAS_AVAILABLE = True
except ImportError:
    try:
        import tensorflow as tf
        import keras
        from keras import layers, ops, Model
        KERAS_AVAILABLE = True
    except ImportError:
        print("Note: TensorFlow not installed. VAE functions won't work.")
        print("      Install with: pip install tensorflow")
        print("      (Only needed for steps 07-08)")


def _check_keras():
    """Raise an error if TensorFlow/Keras is not available."""
    if not KERAS_AVAILABLE:
        raise ImportError(
            "TensorFlow/Keras is required for the VAE. "
            "Install with: pip install tensorflow"
        )


# ============================================================================
#  Sampling layer (reparameterisation trick)
# ============================================================================

class Sampling(layers.Layer):
    """Reparameterisation trick: z = mean + exp(0.5 * log_var) * epsilon.

    This allows gradients to flow through the sampling step.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


# ============================================================================
#  Build the VAE
# ============================================================================

def build_vae(original_dim=6724, intermediate_dim=1028, latent_dim=2, beta=1.0):
    """Build the encoder, decoder, and full VAE models.

    Parameters
    ----------
    original_dim : int
        Size of the flattened FC input (82 × 82 = 6724).
    intermediate_dim : int
        Hidden layer size (default: 1028).
    latent_dim : int
        Latent space dimensions (default: 2).
    beta : float
        Weight on the KL divergence term (default: 1.0).

    Returns
    -------
    (vae, encoder, decoder) : tuple of keras Models
    """
    _check_keras()

    # --- Encoder ---
    inputs = layers.Input(shape=(original_dim,), name="encoder_input")
    h = layers.Dense(intermediate_dim, activation="relu",
                     name="encoder_hidden")(inputs)
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
    z = Sampling(name="z")([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # --- Decoder ---
    latent_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    dh = layers.Dense(intermediate_dim, activation="relu",
                      name="decoder_hidden")(latent_inputs)
    outputs = layers.Dense(original_dim, activation="sigmoid",
                           name="decoder_output")(dh)
    decoder = Model(latent_inputs, outputs, name="decoder")

    # --- Full VAE (custom train_step for the composite loss) ---
    # In Keras 3, add_loss on symbolic tensors no longer works,
    # so we use a Model subclass with custom train_step/test_step.
    # The loss matches the paper (and the original notebook):
    #   L = binary_crossentropy(x, x_decoded) * original_dim
    #       + beta * (-0.5 * sum(1 + log_var - mean^2 - exp(log_var)))
    class VAE(Model):
        def __init__(self, encoder, decoder, original_dim, beta=1.0,
                     **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.original_dim = original_dim
            self.beta = beta
            self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
            self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
            self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        @property
        def metrics(self):
            return [self.total_loss_tracker, self.recon_loss_tracker,
                    self.kl_loss_tracker]

        def call(self, inputs, training=False):
            z_mean, z_log_var, z = self.encoder(inputs)
            return self.decoder(z)

        def _compute_losses(self, x):
            z_mean, z_log_var, z = self.encoder(x)
            x_decoded = self.decoder(z)

            # Reconstruction loss (same as notebook):
            # binary_crossentropy averages over features → multiply
            # by original_dim to get the sum equivalent.
            recon_loss = (
                keras.losses.binary_crossentropy(x, x_decoded)
                * self.original_dim
            )

            # KL divergence (same as notebook)
            kl_loss = -0.5 * ops.sum(
                1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
                axis=-1,
            )

            return recon_loss, kl_loss

        def train_step(self, data):
            x = data[0] if isinstance(data, tuple) else data
            import tensorflow as tf
            with tf.GradientTape() as tape:
                recon_loss, kl_loss = self._compute_losses(x)
                total_loss = ops.mean(recon_loss + self.beta * kl_loss)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.recon_loss_tracker.update_state(ops.mean(recon_loss))
            self.kl_loss_tracker.update_state(ops.mean(kl_loss))
            return {m.name: m.result() for m in self.metrics}

        def test_step(self, data):
            x = data[0] if isinstance(data, tuple) else data
            recon_loss, kl_loss = self._compute_losses(x)
            total_loss = ops.mean(recon_loss + self.beta * kl_loss)
            self.total_loss_tracker.update_state(total_loss)
            self.recon_loss_tracker.update_state(ops.mean(recon_loss))
            self.kl_loss_tracker.update_state(ops.mean(kl_loss))
            return {m.name: m.result() for m in self.metrics}

    vae = VAE(encoder, decoder, original_dim, beta=beta, name="vae")
    vae.compile(optimizer="adam")

    return vae, encoder, decoder


# ============================================================================
#  Training
# ============================================================================

def train_vae(fc_matrices, labels=None, latent_dim=2, intermediate_dim=1028,
              batch_size=256, epochs=5, train_split=0.7, seed=42):
    """Train the VAE on FC matrices.

    Parameters
    ----------
    fc_matrices : ndarray, shape (n_samples, n_regions, n_regions)
        FC matrices. Will be flattened and min-max normalised.
    labels : ndarray or None
        Labels for each sample (for downstream analysis).
    latent_dim : int
        Latent space dimensionality (default: 2).
    intermediate_dim : int
        Hidden layer width (default: 1028).
    batch_size : int
        Training batch size (default: 256).
    epochs : int
        Number of training epochs (default: 5).
    train_split : float
        Fraction of data for training (default: 0.7).
    seed : int
        Random seed for the train/test split.

    Returns
    -------
    (vae, encoder, decoder, history, train_idx, test_idx) : tuple
    """
    _check_keras()

    n_samples = fc_matrices.shape[0]
    n_regions = fc_matrices.shape[1]
    original_dim = n_regions * n_regions

    # Flatten FC matrices
    x_all = fc_matrices.reshape(n_samples, -1).astype("float32")

    # Min-max normalise each sample
    x_min = x_all.min(axis=1, keepdims=True)
    x_max = x_all.max(axis=1, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    x_all = (x_all - x_min) / x_range

    # Train/test split
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(n_samples * train_split)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    x_train = x_all[train_idx]
    x_test = x_all[test_idx]

    # Build and train
    vae, encoder, decoder = build_vae(original_dim, intermediate_dim, latent_dim)
    print(f"Training VAE: {n_train} samples, batch_size={batch_size}, "
          f"epochs={epochs}")

    history = vae.fit(
        x_train, None,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None),
    )

    # Ensure the model is built so save_weights works
    vae.build((None, original_dim))

    return vae, encoder, decoder, history, train_idx, test_idx


# ============================================================================
#  Encoding / Decoding helpers
# ============================================================================

def encode(encoder, fc_matrices):
    """Encode FC matrices into the latent space.

    Parameters
    ----------
    encoder : keras Model
    fc_matrices : ndarray, shape (n_samples, n_regions, n_regions)

    Returns
    -------
    (z_mean, z_log_var, z) : tuple of ndarray
    """
    _check_keras()
    n_samples = fc_matrices.shape[0]
    x = fc_matrices.reshape(n_samples, -1).astype("float32")

    # Min-max normalise
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    x = (x - x_min) / x_range

    return encoder.predict(x, verbose=0)


def decode(decoder, z, n_regions=82):
    """Decode latent points back to FC matrices.

    Parameters
    ----------
    decoder : keras Model
    z : ndarray, shape (n_points, latent_dim)
    n_regions : int
        Number of brain regions (default: 82).

    Returns
    -------
    ndarray, shape (n_points, n_regions, n_regions)
    """
    _check_keras()
    flat = decoder.predict(z, verbose=0)
    return flat.reshape(-1, n_regions, n_regions)
