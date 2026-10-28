"""
Ex. No: 8 - IMAGE AUGMENTATION USING GENERATIVE ADVERSARIAL NETWORKS (GANS)
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Dataset (with offline synthetic fallback)
try:
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype("float32") - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=-1)
except Exception as e:
    print(f"MNIST download failed ({e}). Using synthetic samples...")
    X_train = np.random.uniform(-1, 1, (1000, 28, 28, 1)).astype("float32")

print("Dataset Shape:", X_train.shape)

# 2. Generator Network
def build_generator():
    model = Sequential([
        Dense(128 * 7 * 7, input_dim=100),
        LeakyReLU(negative_slope=0.2),
        Reshape((7, 7, 128)),
        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        Conv2D(1, kernel_size=7, activation="tanh", padding="same")
    ])
    return model

# 3. Discriminator Network
def build_discriminator():
    model = Sequential([
        Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=(28, 28, 1)),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.4),
        Conv2D(128, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.4),
        Flatten(),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=["accuracy"])
    return model

# 4. GAN Compilation
generator = build_generator()
discriminator = build_discriminator()
discriminator.trainable = False

gan = Sequential([generator, discriminator])
gan.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# 5. Adversarial Training Loop
epochs = 500
batch_size = 128
latent_dim = 100

for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise, verbose=0)
    
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    if epoch % 50 == 0:
        g_l = g_loss if np.isscalar(g_loss) else g_loss[0]
        print(f"Epoch {epoch:3d} | D Loss: {d_loss[0]:.4f} | D Acc: {d_loss[1]*100:.2f}% | G Loss: {g_l:.4f}")

# 6. Generate & Visualize Synthetic Images
noise = np.random.normal(0, 1, (10, latent_dim))
generated_images = 0.5 * generator.predict(noise, verbose=0) + 0.5

plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_images[i, :, :, 0], cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()

np.save("augmented_images.npy", generated_images)
print("Synthetic images generated successfully.")

