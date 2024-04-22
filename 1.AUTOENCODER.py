import numpy as np
from pathlib import Path
from astropy.io import fits
import io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Input, Model
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, Flatten, Dense, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import keras
from wandb.keras import WandbCallback


# Configuration dictionary
config = dict(
    batch_size=256,
    model_name='U-Net RealBogus',
    epochs=100,
    init_learning_rate=0.0001,
    lr_decay_rate=0.1,
    optimizer='adam',
    loss_fn='mean_squared_error',
    earlystopping_patience=10,
    metrics=[keras.metrics.KLDivergence(), keras.metrics.MeanAbsoluteError(),
                       keras.metrics.MeanAbsolutePercentageError()]
)

import wandb
from wandb.keras import WandbCallback

wandb.init(project='Real Bogus', config=config)



# Example usage:
h5_folder_path = 'D:/STAMP_AD_IMAGES/ProcessedData'  # Update the path to your .h5 files
all_image_paths = [str(path) for path in Path(h5_folder_path).glob('*.h5')]


# Then split these filtered paths into train, validation, and test sets
train_paths, test_val_paths = train_test_split(all_image_paths, test_size=0.3)
val_paths, test_paths = train_test_split(test_val_paths, test_size=0.5)

import numpy as np
import tensorflow as tf

def crop_center(img, cropx, cropy):
    """Crop the image in the center to the specified size."""
    y, x = img.shape  # Adjusted to expect 2D images
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]



import numpy as np
import h5py
import matplotlib.pyplot as plt



from skimage.exposure import rescale_intensity
import numpy as np
import h5py


def normalize_images(img1, img2):
    """Normalize first and second images together."""
    combined = np.stack([img1, img2])  # Combine images
    p2, p98 = np.percentile(combined, (2, 98))
    img1_rescaled = rescale_intensity(img1, in_range=(p2, p98))
    img2_rescaled = rescale_intensity(img2, in_range=(p2, p98))
    return img1_rescaled, img2_rescaled


def normalize_difference_based_on_range(diff_image, img1, img2):
    """Normalize the third image based on the range of the difference of the first two."""
    difference = img1 - img2
    diff_p2, diff_p98 = np.percentile(difference, (2, 98))
    diff_image_rescaled = rescale_intensity(diff_image, in_range=(diff_p2, diff_p98), out_range=(-1, 1))
    return diff_image_rescaled


def image_generator(image_paths, batch_size):
    """Yield batches of preprocessed input images and their corresponding output images."""
    for file_path in image_paths:
        with h5py.File(file_path, 'r') as hf:
            images = hf['images'][:]  # Assuming 'images' dataset shape is (1000, 3, height, width)

        # Process each triplet in the file
        for start_idx in range(0, images.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, images.shape[0])
            batch_triplets = images[start_idx:end_idx]

            batch_inputs = []
            batch_outputs = []
            for triplet in batch_triplets:
                img1, img2, diff_img = [np.nan_to_num(img, nan=0.0) for img in triplet]
                img1_normalized, img2_normalized = normalize_images(img1, img2)
                diff_img_normalized = normalize_difference_based_on_range(diff_img, img1, img2)

                img1_cropped = crop_center(img1_normalized, 48, 48).reshape(48, 48, 1)
                img2_cropped = crop_center(img2_normalized, 48, 48).reshape(48, 48, 1)
                diff_img_cropped = crop_center(diff_img_normalized, 48, 48).reshape(48, 48, 1)

                input_images = np.concatenate([img1_cropped, img2_cropped], axis=-1)
                output_image = diff_img_cropped
                batch_inputs.append(input_images)
                batch_outputs.append(output_image)

            yield np.array(batch_inputs), np.array(batch_outputs)


def get_dataset(image_paths, batch_size=32, is_training=True):
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(image_paths, batch_size),
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, 48, 48, 2], [None, 48, 48, 1])
    )
    if is_training:
        dataset = dataset.shuffle(buffer_size=256).repeat()
    else:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = get_dataset(train_paths, batch_size=256, is_training=True)
val_dataset = get_dataset(val_paths, batch_size=256, is_training=False)




import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import tensorflow as tf


def visualize_dataset(dataset, n_images=2):
    """
    Visualize a few images from a TensorFlow dataset, showing the concatenated inputs
    and output separately. Assumes dataset yields batches of (input_images, output_images).
    """
    for input_images, output_images in dataset.take(1):  # Take 1 batch from the dataset
        for i in range(min(n_images, input_images.shape[0])):  # Iterate over images in the batch
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            # Input Image 1 (e.g., Science image)
            axs[0].imshow(input_images[i, :, :, 0], cmap='gray')  # First channel of the input
            axs[0].title.set_text("Input Image 1")
            axs[0].axis("off")

            # Input Image 2 (e.g., Template image)
            axs[1].imshow(input_images[i, :, :, 1], cmap='gray')  # Second channel of the input
            axs[1].title.set_text("Input Image 2")
            axs[1].axis("off")

            # Output Image (e.g., Difference image)
            axs[2].imshow(output_images[i, :, :, 0], cmap='gray')  # Output image
            axs[2].title.set_text("Output Image")
            axs[2].axis("off")

            plt.show()


# Visualize images from the training dataset
# visualize_dataset(train_dataset, n_images=7)  # Visualize 3 triplets from the first batch


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, concatenate, \
    BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Activation, Conv2DTranspose
from tensorflow.keras.models import Model

def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Activation('relu')(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Activation('relu')(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Activation('relu')(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)  # Adjusted for simplicity
    c4 = Activation('relu')(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    bn = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    bn = Activation('relu')(bn)
    bn = Dropout(0.3)(bn)
    bn = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn)
    bn = Activation('relu', name='bottleneck')(bn)

    # Decoder
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Activation('relu')(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = Activation('relu')(c6)

    # Additional decoding layers to mirror the encoder structure
    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Activation('relu')(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = Activation('relu')(c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Activation('relu')(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = Activation('relu')(c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Activation('relu')(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = Activation('relu')(c9)

    # Output layer to reconstruct the image
    outputs = Conv2D(input_shape[2], (3, 3), activation='linear', padding='same')(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name="Custom_Autoencoder")
    return model



# Adjust the input shape for your 48x48 images with 2 channels (concatenated images)
input_shape = (48, 48, 2)  # Adjusted for the actual size of your images
model = build_autoencoder(input_shape)
model.summary()


#################################
# Compile Model
#################################

# Callbacks remain unchanged except for Wandb callbacks, which are removed

# Compile and Train Model
opt = tf.keras.optimizers.Adam(learning_rate=config['init_learning_rate'])

# Assuming `model` is your model instance
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=[keras.metrics.KLDivergence(), keras.metrics.MeanAbsoluteError(),
                       keras.metrics.MeanAbsolutePercentageError(), keras.metrics.MeanSquaredError()])  # Include custom MRE metric



# Configure the early stopping and learning rate scheduler callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=config['earlystopping_patience'], restore_best_weights=True
)


def lr_scheduler(epoch, lr):
    # log the current learning rate onto W&B
    if wandb.run is None:
        raise wandb.Error("You must call wandb.init() before WandbCallback()")
    wandb.log({'learning_rate': lr}, commit=False)
    if epoch < 7:
        return lr
    else:
        return lr * tf.math.exp(-config['lr_decay_rate'])

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

opt = tf.keras.optimizers.Adam(learning_rate=config['init_learning_rate'])


# Initialize the W&B run


# Define WandbCallback for experiment tracking
wandb_callback = WandbCallback(
                           monitor='val_loss',
                           log_weights=True,
                           log_evaluation=True)#,
                           #validation_steps=5)


# callbacks
callbacks = [early_stop, wandb_callback, lr_callback]




batch_size = 256

# Calculate the steps per epoch and validation steps
steps_per_epoch = int(len(train_paths)*1000 // (batch_size))
validation_steps = int(len(val_paths)*1000 // (batch_size))

print(steps_per_epoch)
print(validation_steps)



# Train the model
history = model.fit(train_dataset,
                    epochs=100,  # Adjust based on the complexity of the model and the dataset size
                    steps_per_epoch=steps_per_epoch,
                    validation_data=val_dataset,
                    validation_steps=validation_steps,
                    callbacks=callbacks,
                    )




import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.show()




def get_test(image_paths, batch_size=32):
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(image_paths, batch_size),
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, 48, 48, 2], [None, 48, 48, 1])
    )
    return dataset

test_dataset = get_test(test_paths, batch_size=256)

# Evaluate the model on the entire test dataset
test_loss, test_acc = model.evaluate(test_dataset, steps=len(test_paths)//batch_size)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

# Fetch a batch of data from the test dataset
for test_input, test_target in test_dataset.take(2):  # Taking only one batch
    # Generate predictions for this batch
    predictions = model.predict(test_input)

    # Plotting the first N test inputs, their predicted difference images, and the actual difference images
    N = 4  # Number of samples to display
    fig, axs = plt.subplots(N, 4, figsize=(20, 5 * N))  # Adjusted for four images per row
    for i in range(N):
        # Displaying the New image (first channel of the input)
        axs[i, 0].imshow(test_input[i, :, :, 0], cmap='gray')
        axs[i, 0].set_title("New Image")
        axs[i, 0].axis('off')

        # Displaying the Reference image (second channel of the input)
        axs[i, 1].imshow(test_input[i, :, :, 1], cmap='gray')
        axs[i, 1].set_title("Reference Image")
        axs[i, 1].axis('off')

        # Displaying the Actual Difference image (target)
        axs[i, 2].imshow(tf.squeeze(test_target[i]), cmap='gray')
        axs[i, 2].set_title("Actual Difference")
        axs[i, 2].axis('off')

        # Displaying the Predicted Difference image
        axs[i, 3].imshow(predictions[i].squeeze(), cmap='gray')
        axs[i, 3].set_title("Predicted Difference")
        axs[i, 3].axis('off')

    plt.show()
    plt.tight_layout()


wandb.finish()
