import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os


model = tf.keras.models.load_model('model.h5')

count_dog = 0  
count_cat = 0  
total_images = 0  
correct_predictions = 0 


def predict_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    prediction = model.predict(img_array)
    return prediction


folder_path = 'pet_pic/cats'  


for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if file_path.lower().endswith(('.jpg', '.png')):
        total_images += 1 
        print(f"Predicting image: {filename}")
        
        result = predict_image(file_path)
        
        if result[0] > 0.5:  
            print(f"{filename}: Predicted as Class 1 (It is a dog)")
            count_dog += 1
            #correct_predictions += 1  
        else:
            print(f"{filename}: Predicted as Class 0 (It is a cat)")
            count_cat += 1
            correct_predictions += 1 


print("Dog =", count_dog)
print("Cat =", count_cat)


if total_images > 0:
    accuracy = (correct_predictions / total_images) * 100
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No images found in the folder.")
