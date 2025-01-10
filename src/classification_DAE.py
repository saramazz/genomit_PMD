#Implementare codice per addestramento e ottimizzazione di denoising Autoencoder e addestrarlo sui 714 pazienti di training della versione corrente

from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

# Assuming X_train is your training data matrix
input_dim = X_train.shape[1]

# Define the architecture of the autoencoder
input_layer = Input(shape=(input_dim,))
# Add Gaussian noise to inputs
noisy_input = GaussianNoise(0.1)(input_layer)

# Encoder
encoded = Dense(128, activation='relu')(noisy_input)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# Decoder
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Define the model
autoencoder = Model(inputs=input_layer, outputs=decoded)

# Compile the model
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the autoencoder
autoencoder.fit(
    X_train,
    X_train,  # Training on the input to predict the input (denoising)
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.2,  # Use a portion of training data for validation
    verbose=2
)

# Save the trained model if required
autoencoder.save('denoising_autoencoder.h5')