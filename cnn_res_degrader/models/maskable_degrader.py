import tensorflow as tf
from tensorflow import keras


class MaskableDegrader(keras.Model):
    def __init__(self, name: str):
        super(MaskableDegrader, self).__init__(name=name)

    @tf.function
    def train_step(self, data):
        x, y, y_mask = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if self._use_lr_masks:
                y_pred = tf.boolean_mask(y_pred, y_mask)
                y = tf.boolean_mask(y, y_mask)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y, y_mask = data
        y_pred = self(x, training=False)
        if self._use_lr_masks:
            y_pred = tf.boolean_mask(y_pred, y_mask)
            y = tf.boolean_mask(y, y_mask)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def get_functional(self) -> keras.Model:
        x = keras.Input(shape=self._input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))
