import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

CACHE_SIZE = 10  
HISTORY_LENGTH = 50  

# Read dataset
df = pd.read_csv('cache_simulation_data.csv')

# Prepare data
def prepare_data(df, cache_size, history_length):
    num_features = cache_size + history_length
    X = df.drop('label', axis=1).values
    y = df['label'].values

    label_encoder = {label: idx for idx, label in enumerate(np.unique(y))}
    y = np.vectorize(label_encoder.get)(y)

    return X, y

X, y = prepare_data(df, CACHE_SIZE, HISTORY_LENGTH)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

model = Sequential([
    Dense(128, input_dim=CACHE_SIZE + HISTORY_LENGTH),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.3),
    Dense(64),
    LeakyReLU(),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val),
                    class_weight=class_weights, callbacks=[early_stopping, model_checkpoint])

# Evaluate the model on the test set
best_model = tf.keras.models.load_model('best_model.keras')
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

predictions = best_model.predict(X_test[:20])
predicted_labels = np.argmax(predictions, axis=1)
print(f"Predicted labels: {predicted_labels}")

print(X_test[:20])

