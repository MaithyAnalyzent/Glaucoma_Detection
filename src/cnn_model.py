from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def build_cnn_classifier(input_shape=(256, 256, 3), num_classes=2):
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, 3, activation="relu", padding="same"),
            MaxPooling2D(2),
            Conv2D(64, 3, activation="relu", padding="same"),
            MaxPooling2D(2),
            Conv2D(128, 3, activation="relu", padding="same"),
            MaxPooling2D(2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ],
        name="standard_cnn_classifier",
    )
    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
