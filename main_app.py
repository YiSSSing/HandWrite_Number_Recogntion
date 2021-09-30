"""
Usage:
If you want to use this application, you should use
python train_model.py
to build the recognition model first.

"""

import os
import numpy
import argparse
import tkinter
import tensorflow

from PIL import Image, ImageDraw


artboard = None
bool_isMouseLBDown = False

# buttons
btn_recognition = None
btn_saveText = None
btn_clearCanvas = None

# text view
text_ReturnResult = None

# hand-written image
output_img = Image.new("RGB", (336, 336), (255, 255, 255))
draw = ImageDraw.Draw(output_img)

# for saving image
EndPoint = (0, 0)
pic_save_path = 'image/'
last_saved_image_path = None


# here for recognition
model_path = 'model/model.ckpt'
emnist_pic_size = 784  # 28 * 28
int_number_classes = 62  # 26 * 2 + 10
placeholder_x = tensorflow.compat.v1.placeholder("float", [None, emnist_pic_size])
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


# for recognition result
alphabet_set = numpy.arange(62)
for i in range(10, 62):
    alphabet_set[i] += ord('A')


# Hit a plot with LB mouse down event: <Button-1>
def Start_Drawing(event):
    global artboard, EndPoint
    x, y = event.x, event.y
    if x > 400:
        x = 400
    elif x < 10:
        x = 10
    if y > 400:
        y = 400
    elif y < 10:
        y = 10
    EndPoint = (x, y)
    # print("x = " + str(x) + ", y = " + str(y))
    x1, y1 = EndPoint[0]-5, EndPoint[1]-5
    x2, y2 = EndPoint[0]+5, EndPoint[1]+5
    artboard.create_oval(x1, y1, x2, y2, fill='black')


# Drawing line with mouse moving event: <B1-Motion>
def Drawing(event):
    global artboard, EndPoint, output_img, draw
    x, y = event.x, event.y
    if x > 330:
        x = 330
    elif x < 10:
        x = 10
    if y > 330:
        y = 330
    elif y < 10:
        y = 10
    artboard.create_line(EndPoint[0], EndPoint[1], x, y, fill='black', width=12)

    # Create same line on output target image
    coordinate = [EndPoint, (x, y)]
    draw.line(coordinate, (0, 0, 0), width=12)
    EndPoint = (x, y)
    # print("x = " + str(x) + ", y = " + str(y))


def Save_image():
    global artboard, output_img, last_saved_image_path

    int_fname = 1
    while os.path.isfile(pic_save_path + str(int_fname) + '.jpg'):
        int_fname += 1
    filename = str(int_fname) + '.jpg'
    last_saved_image_path = pic_save_path + str(int_fname) + '.jpg'

    os.chdir(pic_save_path)
    output_img.save(filename)
    os.chdir('../')


def Clear_canvas():
    global artboard, output_img, draw
    artboard.delete('all')
    output_img = Image.new("RGB", (336, 336), (255, 255, 255))
    draw = ImageDraw.Draw(output_img)


def Recognition():
    global text_ReturnResult, last_saved_image_path
    Save_image()
    target = Convert_to_Mnist(last_saved_image_path)

    with tensorflow.compat.v1.Session() as session:
        session.run(tensorflow.compat.v1.global_variables_initializer())
        tensorflow.compat.v1.train.Saver().restore(session, model_path)
        result = tensorflow.argmax(output_function, 1)
        result = session.run(result, feed_dict={placeholder_x: target, placeholder_keepingProb: 1})
        result = "Recognition result : " + Convert_Class_to_Text(result[0])

    text_ReturnResult.config(text=result)


def Convert_to_Mnist(img_path):
    img = Image.open(img_path)
    adjust_img = img.resize((28, 28), Image.ANTIALIAS)

    img_array = numpy.array(adjust_img.convert('L'))
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


def main(args):
    global artboard
    global btn_clearCanvas, btn_saveText, btn_recognition
    global text_ReturnResult

    window_main = tkinter.Tk()
    window_main.title('Text Recognition')
    window_main.geometry("480x560")

    artboard = tkinter.Canvas(window_main, width=336, height=336, bg='white', bd=5, highlightbackground='gray')
    artboard.pack(pady=30)
    artboard.bind("<Button-1>", Start_Drawing)  # click on mouse LB
    artboard.bind("<B1-Motion>", Drawing)  # moving the mouse

    btn_recognition = tkinter.Button(window_main, width=12, height=1, text='Recognition', command=Recognition)
    btn_recognition.place(relx=0.1, rely=0.72)

    btn_saveText = tkinter.Button(window_main, width=12, height=1, text='Save', command=Save_image)
    btn_saveText.place(relx=0.4, rely=0.72)

    btn_clearCanvas = tkinter.Button(window_main, width=12, height=1, text='Clear', command=Clear_canvas)
    btn_clearCanvas.place(relx=0.7, rely=0.72)

    text_ReturnResult = tkinter.Label(window_main, text="", font=('Arial', 12), width=30, height=2)
    text_ReturnResult.place(relx=0.2, rely=0.85)

    window_main.mainloop()
    """
    while True:
        cv2.imshow("Text Recognition", artboard)

        # Press Esc or click X on window to close the program
        if (cv2.waitKey(1) == 27) or (cv2.getWindowProperty("Text Recognition", cv2.WND_PROP_VISIBLE) < 1):
            break
    cv2.destroyAllWindows()
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Drawing pad for text recognition.")
    main(parser.parse_args())
