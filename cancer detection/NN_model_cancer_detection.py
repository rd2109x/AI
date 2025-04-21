# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# first 4 lines of output
# /kaggle/input/histopathologic-cancer-detection/sample_submission.csv
# /kaggle/input/histopathologic-cancer-detection/train_labels.csv
# /kaggle/input/histopathologic-cancer-detection/test/a7ea26360815d8492433b14cd8318607bcf99d9e.tif
# /kaggle/input/histopathologic-cancer-detection/test/59d21133c845dff1ebc7a0c7cf40c145ea9e9664.tif

#-------------------------------------------------------------------------------------------------------
file_path = '/kaggle/input/histopathologic-cancer-detection/train_labels.csv'
df = pd.read_csv(file_path)

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumn Data Types:\n", df.dtypes)
#-------------------------------------------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow/CUDA warnings
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # Better T4 utilization
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpu_info = !nvidia-smi
print("".join(gpu_info))
# Should show "Tesla T4" and memory usage
#-------------------------------------------------------------------------------------------------------
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')  # T4 supports FP16 acceleration
mixed_precision.set_global_policy(policy)

# Update your model creation:
def create_model():
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(96, 96, 3))
    
    # Freeze layers more aggressively for T4 memory
    for layer in base_model.layers[:150]:
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', dtype='float32')(x)  # Keep final layers in FP32
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid', dtype='float32')(x)
    
    return Model(inputs=base_model.input, outputs=predictions)
#-------------------------------------------------------------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load and prepare data
labels_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/train_labels.csv')
labels_df['id'] = labels_df['id'] + '.tif'  # Add file extension

# 2. Split into train/validation
train_df, valid_df = train_test_split(
    labels_df,
    test_size=0.2,
    random_state=42,
    stratify=labels_df['label']
)

# 3. Define data generators
batch_size = 128  # Optimized for T4 GPU

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation for validation

# 4. Create generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='/kaggle/input/histopathologic-cancer-detection/train',
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=batch_size,
    class_mode='raw',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory='/kaggle/input/histopathologic-cancer-detection/train',
    x_col='id',
    y_col='label',
    target_size=(96, 96),
    batch_size=batch_size,
    class_mode='raw',
    shuffle=False
)

print("Training samples:", len(train_generator.filenames))
print("Validation samples:", len(valid_generator.filenames))
#-------------------------------------------------------------------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(96,96,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)
#-------------------------------------------------------------------------------------------------------
history = model.fit(
    train_generator,
    validation_data=valid_generator,  # Uses your 44K validation images
    epochs=10,
    batch_size=128
)

#model.save('/kaggle/working/cancer_model.h5')  # Saves architecture + weights
#-------------------------------------------------------------------------------------------------------
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Create a DataFrame of test filenames
test_dir = '/kaggle/input/histopathologic-cancer-detection/test'
test_files = [f for f in os.listdir(test_dir) if f.endswith('.tif')]
test_df = pd.DataFrame({'id': test_files})

# 2. Create test generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_dir,
    x_col='id',
    y_col=None,  # No labels
    target_size=(96, 96),
    batch_size=128,
    class_mode=None,  # No labels
    shuffle=False
)

print(f"Found {len(test_generator.filenames)} test images")  # Should match competition test set size
#-------------------------------------------------------------------------------------------------------
import numpy as np

# Get predicted probabilities (0-1)
test_probs = model.predict(test_generator)

# Convert to binary predictions (adjust threshold as needed)
test_preds = (test_probs > 0.3).astype(int)  # Using 0.3 for higher recall
#-------------------------------------------------------------------------------------------------------
# 2. Create submission DataFrame
submission = pd.DataFrame({
    'id': [f.replace('.tif', '') for f in test_generator.filenames],
    'label': test_preds.flatten()
})

# 3. Save to Kaggle's output directory
output_path = '/kaggle/working/submission.csv'
submission.to_csv(output_path, index=False)

# 4. Verify
print("✅ Submission file saved to OUTPUT folder")
print("Prediction distribution:")
print(submission['label'].value_counts(normalize=True))
print("\nFirst 5 predictions:")
print(submission.head())
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------
