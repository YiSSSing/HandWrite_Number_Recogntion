"""
Recognition of all images in Emnist test dataset.
For testing the model.
"""

from __future__ import division

import tensorflow
import numpy
import tensorflow_datasets as DataSets
import argparse


model_path = 'model/model.ckpt'
int_step = 10000

dataset_emnist, emnist_info = DataSets.load(name="emnist", with_info=True, data_dir="data")
dataset_forTest = dataset_emnist['test']
assert isinstance(dataset_forTest, tensorflow.data.Dataset), "ERROR:: Cannot load the training data from Emnist."
assert emnist_info.features['label'].num_classes == 62, \
        "ERROR:: Loading the wrong dataset. This project should use the Emnist ByClass dataset."
print("\nLoad the Emnist successfully.\n")


# some constant
float_learning_rate = 1e-4
int_batch_size = 32
int_buffer_size = 1024
emnist_pic_size = 784  # 28 * 28
int_number_classes = 62  # 26 * 2 + 10


# For training
placeholder_x = tensorflow.compat.v1.placeholder("float", [None, emnist_pic_size])
placeholder_y = tensorflow.compat.v1.placeholder("float", [None, int_number_classes])
placeholder_keepingProb = tensorflow.compat.v1.placeholder("float")
inputImg = tensorflow.reshape(placeholder_x, [-1, 28, 28, 1])


# Convolution 1 : feature extraction
# filter size = 5*5, channel = 1 (grayscale), 32 filters
# hidden_conv1 returns 28*28*32 feature map
# hidden_pool1 convert feature map to 14*14*32 feature map
w1 = tensorflow.Variable(tensorflow.random.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1, dtype="float"))
b1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[32], dtype="float"))
hidden_conv1 = tensorflow.nn.relu(tensorflow.nn.conv2d(inputImg, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
hidden_pool1 = tensorflow.nn.max_pool2d(hidden_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolution 2 : feature extraction
# filter size = 5*5, channel = 32 (from convolution1, 32 = channel * filters = 1 * 32), 64 filters
# hidden_conv2 returns 14*14*64 feature map
# hidden_pool2 convert feature map to 7*7*64 feature map
w2 = tensorflow.Variable(tensorflow.random.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1, dtype="float"))
b2 = tensorflow.Variable(tensorflow.constant(0.1, shape=[64], dtype="float"))
hidden_conv2 = tensorflow.nn.relu(tensorflow.nn.conv2d(hidden_pool1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
hidden_pool2 = tensorflow.nn.max_pool2d(hidden_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# flatt the feature map as 1D array so we can input CNN pipeline (use reshape)
# use 1024 neurons
# use drop out function to avoid over-fitting
w_f1 = tensorflow.Variable(tensorflow.random.truncated_normal(shape=[7*7*64, 1024], stddev=0.1, dtype="float"))
b_f1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[1024], dtype="float"))
hidden_pool2_flat = tensorflow.reshape(hidden_pool2, [-1, 7*7*64])
hidden_f1 = tensorflow.nn.relu(tensorflow.matmul(hidden_pool2_flat, w_f1) + b_f1)
hidden_f1_dropout = tensorflow.nn.dropout(hidden_f1, placeholder_keepingProb)

# output layer, result is sorted by softmax
# softmax transform the result into probabilities
w_f2 = tensorflow.Variable(tensorflow.random.truncated_normal(shape=[1024, int_number_classes], stddev=0.1, dtype="float"))
b_f2 = tensorflow.Variable(tensorflow.constant(0.1, shape=[int_number_classes], dtype="float"))
output_function = tensorflow.nn.softmax(tensorflow.matmul(hidden_f1_dropout, w_f2) + b_f2)


# Convert Emnist image to standard Mnist style
def Convert_to_Mnist(img_array):
    num0 = num255 = 0
    threshold = 100

    for x in range(28):
        for y in range(28):
            if img_array[x][y] > threshold:
                num255 += 1
            else:
                num0 += 1

    if num255 > num0:
        for x in range(28):
            for y in range(28):
                img_array[x][y] = 255 - img_array[x][y]
                if img_array[x][y] < threshold:
                    img_array[x][y] = 0

    result = img_array.reshape((1, 784))
    result = result.astype(numpy.float32)
    result = numpy.multiply(result, 1 / 255)
    return result


def adjust_img(image):
    img_array = numpy.array(image, dtype='float')
    result = img_array.transpose()
    result = numpy.reshape(result, [28, 28])
    result = Convert_to_Mnist(result)
    return result


def Convert_Class_to_Text(int_class):
    if int_class < 10:
        return str(chr(int_class + ord('0')))
    elif (10 <= int_class) and (int_class < 36):
        int_class -= 10
        return str(chr(int_class + ord('A')))
    else:
        int_class -= 36
        return str(chr(int_class + ord('a')))


def Start_recognition():
    with tensorflow.compat.v1.Session() as session:
        assert isinstance(dataset_forTest, tensorflow.data.Dataset), "ERROR:: Training data is missing."
        print("Start session!")
        # Get the input stream from dataset
        batch = dataset_forTest.batch(1).prefetch(tensorflow.data.experimental.AUTOTUNE)
        # Convert Tensor object to numpy array
        batch = DataSets.as_numpy(batch)

        tensorflow.compat.v1.train.Saver().restore(session, model_path)
        f = open('test.txt', 'w')
        f.write('This file record the result of testing the image recognition on emnist dataset for testing.\n')
        int_correct = 0
        int_false = 0
        for i in range(int_step):
            try:
                data = next(batch)
            except StopIteration:
                break
            img = adjust_img(data['image'])
            real_label = Convert_Class_to_Text(data['label'])
            result = tensorflow.argmax(output_function, 1)
            result = session.run(result, feed_dict={placeholder_x: img, placeholder_keepingProb: 1})
            result_label = Convert_Class_to_Text(result[0])
            out = "Image[" + str(i) + "]: Real label is: " + str(real_label) + ", recognition result is: "\
                  + result_label + ", The recognition on this picture is "
            if real_label == result_label:
                out += "CORRECT\n"
                int_correct += 1
            else:
                out += "False\n"
                int_false += 1
            print(out)
            f.write(out)

        accuracy = int_correct / int_step
        print("Number of Correct: ", str(int_correct))
        print("Number of false: ", str(int_false))
        print("Total accuracy: ", str(accuracy))
        f.write("Number of Correct: " + str(int_correct) + '\n')
        f.write("Number of false: " + str(int_false) + '\n')
        f.write("Total accuracy: " + str(accuracy) + '\n')


def main(args):
    global int_step
    if args.step:
        print("Step = ", str(args.step))
        int_step = int(args.step)
    Start_recognition()
    print("Recognition process over.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the emnist model with test dataset.')
    parser.add_argument('--step', '-i', help='Number of steps.')
    main(parser.parse_args())
