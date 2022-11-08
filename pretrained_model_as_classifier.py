
# пример использования ранее обученной модели как классификатора
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
model = VGG16()

model.summary()

image = load_img('dog.jpg', target_size=(224, 224))
image = img_to_array(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)

# предсказываем вероятность попадания в какой-либо из классов
yhat = model.predict(image)

# конвертируем вероятности в названия классов
label = decode_predictions(yhat)

# получаем наибольшую вероятность, то есть самый вероятный результат
label = label[0][0]


# вывод результата
print('%s (%.2f%%)' % (label[1], label[2]*100))


