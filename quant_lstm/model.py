from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

def create_binary_categorical_model(timesteps, features, units, dropout_rate, learning_rate, l2_lambda=0.001, verbose=False):
    """
    Build a simple LSTM-based binary classifier.

    Parameters
    ----------
    timesteps : int
        Number of timesteps in each input sequence.
    features : int
        Number of features (columns) at each timestep.
    units : int
        Number of hidden units in the LSTM layer.
    dropout_rate : float
        Dropout fraction (e.g., 0.2 for 20% dropout).
    learning_rate : float
        Learning rate for the Adam optimizer.
    l2_lambda : float, default 0.001
        L2 regularization coefficient.

    Returns
    -------
    tf.keras.Model
        Compiled binary-classification model.
    """
    # L2 regularizer
    l2 = regularizers.l2(l2_lambda)

    model = Sequential([
        Input(shape=(timesteps, features)),
        LSTM(units, kernel_regularizer=l2),
        Dense(32, activation='relu', kernel_regularizer=l2),
        Dropout(dropout_rate),
        # OUTPUT: single neuron with sigmoid for binary
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    loss = 'binary_crossentropy'
    #loss = binary_focal_loss(alpha=0.6, gamma=2.0)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    if verbose:
        model.summary()

    return model