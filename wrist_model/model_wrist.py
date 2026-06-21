import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, ReLU, MaxPooling1D,
    Flatten, Dropout, Bidirectional, LSTM, Dense, Reshape
)


class WristSleepNet(tf.keras.Model):
    """
    Lightweight 1D-CNN + BiLSTM for wrist-sensor sleep staging.
    Input: engineered features per 30-second epoch (HR, HRV, accel, temp, EDA).
    Output: 4-class logits — Wake / Light / Deep / REM.
    """

    def __init__(self, config, **kwargs):
        super(WristSleepNet, self).__init__(**kwargs)
        self.config = config
        n_features = config["input_size"]   # 12 features per epoch
        l2_reg = tf.keras.regularizers.l2(config["l2_weight_decay"])

        # 1D-CNN feature extractor over the feature vector
        # Treats the 12 features as a 1D "signal" of length 12
        self.conv1 = Conv1D(64, kernel_size=3, padding="same", kernel_regularizer=l2_reg)
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()

        self.conv2 = Conv1D(128, kernel_size=3, padding="same", kernel_regularizer=l2_reg)
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.pool = MaxPooling1D(pool_size=2)

        self.flatten = Flatten()
        self.drop1 = Dropout(0.5)

        # BiLSTM over the sequence of epochs
        self.rnn = Bidirectional(LSTM(64, return_sequences=True))
        self.drop2 = Dropout(0.5)

        # Classifier head
        self.dense1 = Dense(64, activation="relu", kernel_regularizer=l2_reg)
        self.drop3 = Dropout(0.5)
        self.output_layer = Dense(config["n_classes"], name="logits")

    def call(self, inputs, training=False):
        # inputs: (batch * seq_len, n_features)
        # Expand to (batch * seq_len, n_features, 1) for Conv1D
        x = tf.expand_dims(inputs, axis=-1)

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.pool(x)

        x = self.flatten(x)
        features = self.drop1(x, training=training)

        # Reshape flat features into (batch, seq_len, feature_dim)
        seq_length = self.config["seq_length"]
        batch_dim = tf.shape(features)[0]
        feature_dim = features.shape[-1]

        if batch_dim % seq_length == 0:
            batch_size = batch_dim // seq_length
            features_seq = tf.reshape(features, (batch_size, seq_length, feature_dim))
        else:
            features_seq = tf.reshape(features, (1, -1, feature_dim))

        rnn_out = self.rnn(features_seq, training=training)
        rnn_out = self.drop2(rnn_out, training=training)

        # Flatten back to (batch * seq_len, rnn_units * 2)
        rnn_flat = tf.reshape(rnn_out, (-1, 128))   # 64 units * 2 directions

        x = self.dense1(rnn_flat)
        x = self.drop3(x, training=training)
        logits = self.output_layer(x)

        return logits
