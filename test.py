from keras.models import load_model
from os import listdir
from os.path import join, isdir
from keras.preprocessing import image
import numpy as np

path = '/Volumes/ADATA HD330/hw2/02468'
model = load_model('CNN.h5')
dir = listdir(path)
dir.sort()
f = open('./Answer.txt', 'w')

for i in range(5000):
    input = join(path, dir[i])
    img = image.load_img(input, target_size = (28, 28))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.
    test = model.predict_classes(x)
    f.write(dir[i][:-4] + ' ' + str(test[0] * 2) + '\n')
f.close()