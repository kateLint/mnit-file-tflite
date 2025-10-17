import math
import os

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

tf.random.set_seed(42)
BATCH_SIZE = 128
(ds_train_data, ds_val_data), info = tfds.load(
    name='mnist',
    split=['train', 'test'],
    with_info=True,
    as_supervised=True,
)

num_classes = info.features['label'].num_classes
def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(label, tf.int32)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

ds_train = (
    ds_train_data
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(info.splits['train'].num_examples)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

ds_val = (
    ds_val_data
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTOTUNE)
)
inputs = layers.Input(shape=(28, 28, 1), name='input')

x = layers.Conv2D(24, kernel_size=(6, 6), strides=1)(inputs)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv2D(48, kernel_size=(5, 5), strides=2)(x)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv2D(64, kernel_size=(4, 4), strides=2)(x)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Flatten()(x)
x = layers.Dense(200)(x)
x = layers.BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(rate=0.25)(x)

predictions = layers.Dense(num_classes, activation='softmax', name='output')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
lr_decay = lambda epoch: 0.0001 + 0.02 * math.pow(1.0 / math.e, epoch / 3.0)
decay_callback = LearningRateScheduler(lr_decay, verbose=1)

model.fit(
    ds_train,
    epochs=20,
    validation_data=ds_val,
    callbacks=[decay_callback],
    verbose=1
)

# Freeze BatchNorm and create an inference-only concrete function to avoid MLIR issues
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
model.trainable = False

run_inference = tf.function(lambda x: model(x, training=False))
concrete_func = run_inference.get_concrete_function(
    tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.float32)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
try:
    tflite_model = converter.convert()
except Exception as convert_error:
    print(f"Standard TFLite conversion failed: {convert_error}\nRetrying with Select TF ops...")
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()

script_dir = os.path.dirname(os.path.abspath(__file__))
tflite_path = os.path.join(script_dir, 'mnist.tflite')
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved to: {tflite_path}")