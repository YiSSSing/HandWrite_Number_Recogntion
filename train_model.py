"""
Training hand-written recognition using emnist dataset by CNN.

Training Accuracy: 86%
"""

import tensorflow
import tensorflow_datasets as DataSets
import numpy


__all__ = [tensorflow]
mnist = tensorflow.keras.datasets.mnist

# dataset and resources
dataset_emnist, emnist_info = None, None
dataset_forTrain = None
model_path = 'model/model.ckpt'


# some constant
float_learning_rate = 1e-4
int_number_steps = 50000
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


def Load_Dataset():
    global dataset_emnist, emnist_info
    global dataset_forTrain
    dataset_emnist, emnist_info = DataSets.load(name="emnist", with_info=True, data_dir="data")
    dataset_forTrain = dataset_emnist['train']
    assert isinstance(dataset_forTrain, tensorflow.data.Dataset), "ERROR:: Cannot load the training data from Emnist."
    assert emnist_info.features['label'].num_classes == 62, \
        "ERROR:: Loading the wrong dataset. This project should use the Emnist ByClass dataset."
    print("\nLoad the Emnist successfully.\n")


# The class label in Emnist dataset is an integer number.
# This function returns the corresponding numpy array.
def ConvertClassToArray(int_class):
    nparray = numpy.zeros(int_number_classes)
    nparray[int_class] = 1
    return nparray


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


# Get the batched data.
def next_batch(generator):
    img_array = numpy.zeros((int_batch_size, emnist_pic_size))
    label_array = numpy.zeros((int_batch_size, int_number_classes))
    for i in range(0, int_batch_size):
        data = next(generator)

        # Images in Emnist dataset are rotated 90 degrees, need to rotate back (array transpose).
        img = numpy.array(data['image'], dtype='float')
        img = img.transpose()

        img = numpy.reshape(img, [28, 28])
        img = Convert_to_Mnist(img)

        img_array[i] = img
        label_array[i] = ConvertClassToArray(data['label'])

    return img_array, label_array


def Start_Training():
    with tensorflow.compat.v1.Session() as session:
        assert isinstance(dataset_forTrain, tensorflow.data.Dataset), "ERROR:: Training data is missing."
        # define parameters for training
        cross_entropy = -tensorflow.reduce_sum(placeholder_y * tensorflow.math.log(output_function))
        optimize = tensorflow.compat.v1.train.AdamOptimizer(learning_rate=float_learning_rate)
        train_op = optimize.minimize(cross_entropy)

        correct_pred = tensorflow.equal(tensorflow.argmax(placeholder_y, 1), tensorflow.argmax(output_function, 1))
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_pred, "float"))

        session.run(tensorflow.compat.v1.global_variables_initializer())
        float_total_accuracy = 0

        # Get the input stream from dataset
        batch = dataset_forTrain.repeat().shuffle(int_buffer_size).batch(1).prefetch(tensorflow.data.experimental.AUTOTUNE)
        # Convert Tensor object to numpy array
        batch = DataSets.as_numpy(batch)

        f = open('log.txt', 'w')
        for i in range(1, int_number_steps+1):
            img, label = next_batch(batch)
            session.run(train_op, feed_dict={placeholder_x: img, placeholder_y: label, placeholder_keepingProb: 0.5})
            acc = session.run(accuracy, feed_dict={placeholder_x: img, placeholder_y: label, placeholder_keepingProb: 1})
            float_total_accuracy += float(acc)
            string = 'Step: ' + str(i) + ', Accuracy: ' + str(acc) + '\n'
            f.write(string)
            print("Step: " + str(i) + " , Accuracy: " + str(acc))

        print("Training completed.")
        print("Average accuracy: ", float_total_accuracy / int_number_steps)

        try:
            save_path = tensorflow.compat.v1.train.Saver().save(session, model_path)
            print("Model have saved successfully to: ", save_path)
        except Exception as e:
            print(e)
            print("Cannot save the model.")


def main():
    # print(DataSets.list_builders())
    Load_Dataset()
    Start_Training()
    print("Training Finished.")


if __name__ == '__main__':
    main()
