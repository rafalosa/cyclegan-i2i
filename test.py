import tensorflow as tf


class TestModel(tf.keras.models.Model):

    def __init__(self):
        super(TestModel, self).__init__()

        self.internal_model_1 = tf.keras.models.Sequential([

            tf.keras.layers.Dense(100, input_dim=(60, 60)),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(5, activation="relu")
        ])

        self.internal_model_2 =  tf.keras.models.Sequential([

            tf.keras.layers.Dense(120, input_dim=(60, 60)),
            tf.keras.layers.Dense(60, activation="relu"),
            tf.keras.layers.Dense(2, activation="relu")
        ])

    def call(self, inputs, training=None, mask=None):
        pass


if __name__ == "__main__":

    model = TestModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=.001, beta_1=.5)

    # test 1, total model
    checkpoint = tf.train.Checkpoint(total_model=model)

    # # test 2, partial checkpoint
    # checkpoint = tf.train.Checkpoint(internal1=model.internal_model_1,
    #                                  internal2=model.internal_model_2)

    model.compile(optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory="checkpoint_test_dir", max_to_keep=3)
