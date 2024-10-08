import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


grayscale_dir = 'flowers_grey'
color_dir = 'flowers_colour'

image_size = (128, 128)


grayscale_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    grayscale_dir,
    label_mode=None,
    color_mode='grayscale',
    image_size=image_size,
    batch_size=32,
shuffle = False
)


color_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    color_dir,
    label_mode=None,
    color_mode='rgb',
    image_size=image_size,
    batch_size=32,
shuffle=False

)


grayscale_dataset = grayscale_dataset.map(lambda x: x / 255.0)
color_dataset = color_dataset.map(lambda x: x / 255.0)


for gray_batch, color_batch in zip(grayscale_dataset, color_dataset):
    print("Grayscale batch shape:", gray_batch.shape)
    print("Color batch shape:", color_batch.shape)
    break


def build_generator():
    inputs = layers.Input(shape=(128, 128, 1))

    # Encoder (downsampling)
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Decoder (upsampling)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')(x)

    return models.Model(inputs=inputs, outputs=x)

generator = build_generator()
generator.summary()


loss_object = tf.keras.losses.MeanSquaredError()


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


@tf.function
def train_step(input_image, target_image):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)
        gen_loss = loss_object(target_image, gen_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    return gen_loss


def train(dataset, epochs):
    for epoch in range(epochs):
        for input_image, target_image in dataset:
            gen_loss = train_step(input_image, target_image)

        print(f"Epoch {epoch + 1}, Generator Loss: {gen_loss}")


# Combine datasets
dataset = tf.data.Dataset.zip((grayscale_dataset, color_dataset))

# Train the model
train(dataset, epochs=50)

# Save the trained model
generator.save('grayscale_to_color_model.h5')
