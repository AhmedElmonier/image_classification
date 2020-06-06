import img_class
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np

new_image = plt.imread('4.jpg')
img = plt.imshow(new_image)
#plt.show()

resized_image = resize(new_image, (32,32,3))
img = plt.imshow(new_image)
#plt.show()

predictions = img_class.model.predict(np.array([resized_image]))
print(predictions)



list_index = [0,1,2,3,4,5,6,7,8,9]
x = predictions

for i in range(10):
    for j in range(10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp


print(list_index)

for i in range(5):
    print(img_class.classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')

