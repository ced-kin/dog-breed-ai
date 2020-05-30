# dog-breed-ai  
The goal of this project was to produce an image classifier to determine the breed of a dog and implement that classifier in a mobile application. 

## CNN from scratch     
Conv2D(16, 3, padding='same', activation='relu', input_shape=(180, 180, 3))  
Dropout(0.1)  
MaxPooling2D()  
Conv2D(32, 3, padding='same', activation='relu')  
MaxPooling2D()  
Conv2D(64, 3, padding='same', activation='relu')  
MaxPooling2D()  
Flatten()  
Dense(512, activation='relu')  
Dropout(0.5)  
Dense(120)  
  
After 10 epochs this network achieved an accuracy of 13%. The dataset has 12,000 images for training. This averages to 100 images per class. However, even with data augmentation this doesn't appear to be enough data. 

## Transfer learning  
The Xception model was used as a feature extractor.  

xception_model  
GlobalAveragePooling2D()  
Dense(120)  
  
Note: the xception model was not trainable and had input_shape=(299, 299, 3)  
After 10 epochs this network achieved an accuracy of 88%.  

## Mobile app  
The model was turned into a tflite file and used to develop the android application. The app allows you to take a picture which is then saved and processed by the classifier. The results of the classifier are displayed on a separate screen. The whole process takes about 1 second to complete (time between pressing 'take picture' and seeing the results screen).

## Result
An android application that classifies 120 dog breeds with 88% accuracy

## Acknowledgements
Transfer learning - https://www.tensorflow.org/tutorials/images/transfer_learning  
Xception model - https://arxiv.org/abs/1610.02357  
Stanford Dogs Dataset - http://vision.stanford.edu/aditya86/ImageNetDogs/  
Android CameraX - https://developer.android.com/training/camerax  
Android image classification - https://heartbeat.fritz.ai/image-classification-on-android-with-tensorflow-lite-and-camerax-4f72e8fdca79

