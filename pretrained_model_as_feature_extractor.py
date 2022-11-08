
# пример использования модели VGG16 как модели препроцессинга для извлечения признаков
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump


image = load_img('dog.jpg', target_size=(224, 224))

image = img_to_array(image)

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

image = preprocess_input(image)

model = VGG16()
# удаляем выходной слой
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
# получаем извлеченные признаки
features = model.predict(image)
print(features.shape)
# сохраняем в файл
dump(features, open('dog.pkl', 'wb'))

