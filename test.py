

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'pet_pic',      
    target_size=(150, 150),  
    batch_size=32,
    class_mode='binary'      
)

model = load_model('model.h5')

test_loss, test_accuracy = model.evaluate(test_set)
print(f'Test accuracy: {test_accuracy},{test_accuracy*100:.2f}%')