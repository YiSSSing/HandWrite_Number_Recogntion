"""
This program is for watching the source image of emnist.
"""


import numpy
import tensorflow
import tensorflow_datasets as DataSets
from matplotlib import pyplot


dataset_emnist, emnist_info = DataSets.load(name="emnist", with_info=True, data_dir="data")
dataset_forTrain, dataset_forTest = dataset_emnist['train'], dataset_emnist['test']
assert isinstance(dataset_forTrain, tensorflow.data.Dataset), "ERROR:: Cannot load the training data from Emnist."
assert emnist_info.features['label'].num_classes == 62, \
        "ERROR:: Loading the wrong dataset. This project should use the Emnist ByClass dataset."
print("Load the Emnist successfully.\n")


def Convert_Class_to_Text(int_class):
    print("Class is: ", int_class)
    if int_class < 10:
        return str(int_class)
    elif (10 <= int_class) and (int_class < 36):
        int_class -= 10
        return str(chr(int_class + ord('A')))
    else:
        int_class -= 36
        return str(chr(int_class + ord('a')))


# Get the input stream from dataset
batch = dataset_forTrain.shuffle(16).batch(1).prefetch(tensorflow.data.experimental.AUTOTUNE)
# Convert Tensor object to numpy array
batch = DataSets.as_numpy(batch)

with tensorflow.Session() as session:
    for i in range(10):
        data = next(batch)

        # Show origin image
        label = Convert_Class_to_Text(data['label'])
        print("origin picture, Label = ", label)
        img = numpy.array(data['image'], dtype='float')
        pixels = img.reshape((28, 28))
        pyplot.imshow(pixels, cmap='gray')
        pyplot.show()

        # Show the image after rotate
        img = img.transpose()
        print("Rotate picture, Label = ", label)
        t = numpy.array(img, dtype='float')
        pixels = t.reshape((28, 28))
        pyplot.imshow(pixels, cmap='gray')
        pyplot.show()
