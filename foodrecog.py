import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
import os

#check right tensorflow version and gpu

print("TensorFlow version:", tf.__version__)
gpu = tf.config.list_physical_devices('GPU')
print("GPU Available:", gpu)

#make/set checkpoint directory

CHECKPOINT_DIR = "/Users/manuelpaul/Downloads/brand_new_food101_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#download data

dataset, info = tfds.load("food101", with_info=True, as_supervised=True)
train_ds, test_ds = dataset["train"], dataset["validation"]

print("Number of classes:", info.features["label"].num_classes)
print("Class names:", info.features["label"].names[:10], "...")

#preprocess data

IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

#build model

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(info.features["label"].num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "new_ckpt_10.weights.h5")
# if os.path.exists(latest_checkpoint_path):
#     print(f"Resuming from {latest_checkpoint_path}")
#     model.load_weights(latest_checkpoint_path)
# else:
#     print("No checkpoint found, starting fresh.")

#save checkpoint after each epoch

checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, "brand_new_ckpt_{epoch:02d}.weights.h5"),
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)

model.summary()

#train model

EPOCHS = 10
INITIAL_EPOCH = 0

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=test_ds,
    callbacks=[checkpoint_cb],
    initial_epoch=INITIAL_EPOCH
)

#unfreeze model + fine tune

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    epochs=INITIAL_EPOCH+EPOCHS+15,  # continue training
    initial_epoch=INITIAL_EPOCH+EPOCHS,
    validation_data=test_ds,
    callbacks=[checkpoint_cb]
)

#evaluate model
loss, acc = model.evaluate(test_ds)
print(f"Test accuracy: {acc:.4f}")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = "new_food101_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"TFLite model saved as {tflite_path}")
