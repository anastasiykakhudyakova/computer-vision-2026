'''сделать полносвязную и сверточную нейросеть на датасете svhn cropped без использования готовой модели'''
import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt

print("Loading SVHN cropped...")
(ds_train, ds_test), ds_info = tfds.load(
    'svhn_cropped',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, depth=10)
    return image, label

batch_size = 64
ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_size = ds_info.splits['train'].num_examples
val_size = int(0.2 * train_size)

ds_train_unbatched = (
    ds_train
    .map(preprocess)
    .unbatch()
)

ds_val = ds_train_unbatched.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_train_tune = ds_train_unbatched.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print(f"Train: {train_size - val_size}, Val: {val_size}, Test: {ds_info.splits['test'].num_examples}")

def build_cnn(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))

    num_blocks = hp.Int('num_conv_blocks', 1, 3)
    base_filters = 32

    for i in range(num_blocks):
        filters = base_filters * (2 ** i)
        model.add(tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
        model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        drop = hp.Float(f'conv_dropout_{i}', 0.1, 0.5, step=0.1)
        model.add(tf.keras.layers.Dropout(drop))

    model.add(tf.keras.layers.Flatten())

    dense_units = hp.Int('dense_units', 128, 512, step=128)
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    dense_drop = hp.Float('dense_dropout', 0.2, 0.6, step=0.1)
    model.add(tf.keras.layers.Dropout(dense_drop))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.Hyperband(
    build_cnn,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='svhn_tuning',
    project_name='cnn',
    overwrite=True
)

tuner.search(
    ds_train_tune,
    epochs=20,
    validation_data=ds_val,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=4)]
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest CNN hyperparameters:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

ds_full_train = ds_train_unbatched.batch(batch_size).prefetch(tf.data.AUTOTUNE)

best_model = tuner.hypermodel.build(best_hp)
print("\nTraining best CNN model...")
best_model.fit(ds_full_train, epochs=25, validation_data=ds_test, verbose=2)

test_loss, test_acc = best_model.evaluate(ds_test, verbose=0)
print(f"\nCNN test accuracy: {test_acc:.4f}")
