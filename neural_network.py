import tensorflow as tf
import os
import numpy as np
import csv
import random
import time
import hand_extract as he
import cv2
import sys
import string
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Conv1D, Dense, Flatten, MaxPool2D,MaxPool1D, Softmax
from tensorflow.python.tools import inspect_checkpoint as chkp

imgWidth = 200
imgHeight = 200
train_data = list()
des_out = list()
test_data = list()
des_out_test = list()
images_path = 'E:\\Licenta2019\\Sign_Language_translator\\conv_dataset2'
folder_name = 'E:/test/log_at_{}'.format(int(time.time()))
conv_folder_name = 'E:/conv/log_at_{}'.format(int(time.time()))
imgs = list()


def prob_to_letter_conv(predictions):
    highest_val = 0
    
    for x in range(1,8):
        if predictions[x] > predictions[highest_val]:
            highest_val = x

    if highest_val == 0:
        return 'A'
    elif highest_val == 1:
        return 'B'
    elif highest_val == 2:
        return 'C'
    elif highest_val == 3:
        return 'D'
    elif highest_val == 4:
        return 'F'       
    elif highest_val == 5:
        return 'G'
    elif highest_val == 6:
        return 'I'       
    elif highest_val == 7:
        return 'J'                   
            
def prob_to_letter(predictions):
    highest_val = 0
    
    for x in range(1,26):
        if predictions[x] > predictions[highest_val]:
            highest_val = x
        
    if highest_val == 0:
        return 'A'
    elif highest_val == 1:
        return 'B'
    elif highest_val == 2:
        return 'C'
    elif highest_val == 3:
        return 'D'
    elif highest_val == 4:
        return 'F'       
    elif highest_val == 5:
        return 'G'
    elif highest_val == 6:
        return 'I'       
    elif highest_val == 7:
        return 'P'       
    elif highest_val == 8:
        return 'V'       
    elif highest_val == 9:
        return 'W'       
    elif highest_val == 10:
        return 'Y'        
    elif highest_val == 11:
        return 'J'
    elif highest_val == 12:
        return 'L'
    elif highest_val == 13:
        return 'O'
    elif highest_val == 14:
        return 'Q'
    elif highest_val == 15:
        return 'E'
    elif highest_val == 16:
        return 'H'
    elif highest_val == 17:
        return 'R'
    elif highest_val == 18:
        return 'S'
    elif highest_val == 19:
        return 'T'
    elif highest_val == 20:
        return 'U'
    elif highest_val == 21:
        return 'X'
    elif highest_val == 22:
        return 'Z'
    elif highest_val == 23:
        return 'M'
    elif highest_val == 24:
        return 'N'        
    elif highest_val == 25:
        return 'K' 
        
def convert_to_prob(letter):
    if letter == ' A':
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter == ' B':
        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter == ' C':
        return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter == ' D':
        return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter == ' F':
        return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter == ' G':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter == ' I':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter == ' P':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]        
    elif letter == ' V':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter == ' W':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter == ' Y':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter ==  ' J':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter ==  ' L':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter ==  ' O':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif letter ==  ' Q':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    elif letter ==  ' E':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    elif letter ==  ' H':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    elif letter ==  ' R':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    elif letter ==  ' S':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    elif letter ==  ' T':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    elif letter ==  ' U':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    elif letter ==  ' X':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0] 
    elif letter ==  ' Z':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] 
    elif letter ==  ' M':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] 
    elif letter ==  ' N':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]         
    elif letter ==  ' K':
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] 

def convert_to_prob_conv(letter):
        if letter == 'A':
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif letter == 'B':
            return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif letter == 'C':
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif letter == 'D':
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        elif letter == 'F':
            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        elif letter == 'G':
            return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif letter == 'I':
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        elif letter == 'J':
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]            
        
def read_from_csv(file, input_list, label_list):
    x = 0
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        data = list(csv_reader)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)
    for row in data:
        x = x + 1 
        label_list.append( convert_to_prob(row[len(row)-1]) )
        del row[-1]
        for x in range(len(row)):
            row[x] = float(row[x])
        input_list.append(row)

def prepare_dataset(start, end, imgs_list):
    global images_path
    images = list()
    labels = list()
    img_names = list()
    for x in range(start, end):

        img = cv2.imread(imgs_list[x], 1)
#        img = np.expand_dims(img, axis = 2)
        img = cv2.resize(img, (200,200))
        images.append(img/255.0)
        last_app = imgs_list[x].rfind("\\")
        img_name = imgs_list[x][last_app+1:]
        labels.append(convert_to_prob_conv(img_name[0]))  
        img_names.append(imgs_list[x]) 
    return images, labels, img_names                
        
def get_images(path):
    global imgWidth
    global imgHeight
    index = 0
    test_images = list()
    train_images = list()
    
    images = he.get_all_files(path)   
    random.shuffle(images)
    random.shuffle(images)
    random.shuffle(images)

    for image in images:
        if (index % 200) == 0:
            test_images.append(image)
        else:
            train_images.append(image)
        index = index + 1    
    return train_images, test_images

def get_conv_output(train_images):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    img_names = list()
    global imgWidth
    global imgHeight
       
    nrChannels = 3
    nrClasses = 8
    learnRate = 0.01
    
    input = tf.placeholder(shape=[None, imgHeight , imgWidth, nrChannels], dtype=tf.float32, name = "input")
    name = tf.placeholder(dtype = tf.string)
    
    conv1 = tf.layers.conv2d(input, 3, 5, activation=tf.nn.relu, kernel_initializer = tf.initializers.variance_scaling(), name = "conv1")
    conv1_reshaped = tf.image.resize_image_with_crop_or_pad(conv1, imgWidth + 4, imgHeight + 4)
    conv1_summary = tf.summary.image("first image", conv1_reshaped)
    pool1 = tf.layers.max_pooling2d(conv1, 4, 4, name = "pool1")

    conv2 = tf.layers.conv2d(pool1, 3, 5, activation=tf.nn.relu, kernel_initializer = tf.initializers.variance_scaling(), name = "conv2")
    conv2_reshaped = tf.image.resize_image_with_crop_or_pad(conv2, imgWidth + 4, imgHeight + 4)
    conv2_summary = tf.summary.image("second image", conv2_reshaped)
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name = "pool2")


    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter( conv_folder_name + '/train')
    nrEpochs = len(train_images)
    print(nrEpochs)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer() )
        for j in range(0,nrEpochs):
            if j <= nrEpochs - 2:
                data_to_train, _, names = prepare_dataset(j, j+1, train_images)
                last_app = names[0].rfind("\\")
                name1 = names[0][last_app+1:]
                #split_name = name.split(".")
                print(j)
                trainData = { input : data_to_train, name : name1}
                                
                summary = sess.run(merged, feed_dict = trainData)                                
                writer.add_summary(summary, j)
              
                
def define_and_run_conv_net(train_images, test_images):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    img_names = list()
    global imgWidth
    global imgHeight
       
    nrChannels = 3
    nrClasses = 8
    learnRate = 0.0001
    
    input = tf.placeholder(shape=[None, imgHeight , imgWidth, nrChannels], dtype=tf.float32, name = "input")
    out = tf.placeholder(shape=[None, nrClasses], dtype = tf.float32, name = "out")
    
    conv1 = tf.layers.conv2d(input, 32, 5, activation=tf.nn.relu, kernel_initializer = tf.initializers.variance_scaling(), name = "conv1")
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name = "pool1")
    conv2 = tf.layers.conv2d(pool1, 32, 5, activation=tf.nn.relu, kernel_initializer = tf.initializers.variance_scaling(), name = "conv2")
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name = "pool2")    
    conv3 = tf.layers.conv2d(pool2, 32, 5, activation=tf.nn.relu, kernel_initializer = tf.initializers.variance_scaling(), name = "conv3")
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name = "pool3")
    
    flatten = tf.contrib.layers.flatten(pool2)
    tf.summary.scalar('flatten', tf.reduce_mean(flatten) )
    densed1 = tf.layers.dense(flatten, 100, activation=tf.nn.sigmoid,kernel_initializer = tf.initializers.glorot_uniform(), name = "densed1")
    tf.summary.scalar('dense1', tf.reduce_mean(densed1) )
    densed2 = tf.layers.dense(densed1, nrClasses, activation = tf.nn.sigmoid, kernel_initializer = tf.initializers.glorot_uniform(),  name = "densed2")
    tf.summary.scalar('dense2', tf.reduce_mean(densed2) )
    predOuts = tf.nn.softmax(densed2, name = "predOuts")
    tf.summary.scalar('mean_predOuts', tf.reduce_mean(predOuts) )
    tf.summary.scalar('max_predOuts', tf.reduce_max(predOuts) )
    
    #loss = tf.reduce_mean(tf.square(out - predOuts))
    loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = out, logits = predOuts))
    tf.summary.scalar('mean_loss', tf.reduce_mean(loss) ) 
    optimizer = tf.train.AdamOptimizer(learnRate).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(predOuts, 1))
    
    accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))
    tf.summary.scalar('mean_acc', tf.reduce_mean(accuracy) ) 

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter( conv_folder_name + '/test' )    
    train_writer = tf.summary.FileWriter( conv_folder_name + '/train' )
    
    test_data, des_out_test, _ = prepare_dataset(0, len(test_images), test_images)
    nrEpochs = int(len(train_images) / 100 )
    nrRepetitions = 2
    
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer() )
            
        for i in range(nrRepetitions):
            for j in range(nrEpochs):            
                if  j != 0 and j % (nrEpochs - 1) == 0:
                    data_to_train, des_for_train, _ = prepare_dataset(j*100, len(train_images), train_images)
                else:
                    data_to_train, des_for_train, _ = prepare_dataset(j*100, (j+1)*100, train_images)        
                trainData = {input : data_to_train, out : des_for_train}
                opt,current_loss = sess.run([optimizer, loss], feed_dict = trainData)
                eval = sess.run( accuracy, feed_dict = trainData )

                summary = sess.run(merged, feed_dict = trainData)
                train_writer.add_summary(summary, i*nrEpochs + j)
                
                if j % 5 == 0:
                    testData = { input : test_data , out : des_out_test }
                    eval, test_loss = sess.run( [accuracy,loss]  , feed_dict = testData )
                    
                    summary = sess.run(merged, feed_dict = testData)                                
                    test_writer.add_summary(summary, i*nrEpochs + j)
                    print(j)
                    print('accuracy = ', eval )
                    print('current Loss = ', test_loss )
                    print("\n\n")
                if i == nrRepetitions - 1 and j == nrEpochs - 1:
                    saver.save(sess, "conv_model/ConvModel.ckpt")
                    
def define_and_run_neural_net(train_mean, train_stddev):
    global train_data
    global des_out
    global test_data
    global des_out_test
    

    inLayerSize = 44
    hiddenLayerSize1 = 88    
    hiddenLayerSize2 = 132    
    outLayerSize = 26

    inPH = tf.placeholder(tf.float32, [None, inLayerSize], name = "inPH" )
    desOutPH = tf.placeholder(tf.float32, [None, outLayerSize], name = "desOutPH" )
    keep_prob = tf.placeholder(tf.float32)
    
    weights1 = tf.get_variable("weights1", [inLayerSize,hiddenLayerSize1], initializer=tf.contrib.layers.variance_scaling_initializer() )
    biases1 = tf.Variable(0.5, dtype = tf.float32, name="biases1")    
    weights2 = tf.get_variable("weights2", [hiddenLayerSize1, hiddenLayerSize2], initializer=tf.contrib.layers.variance_scaling_initializer() )
    biases2 = tf.Variable(0.5, dtype = tf.float32, name = "biases2")
    weights3 = tf.get_variable("weights3", [hiddenLayerSize2, outLayerSize], initializer=tf.contrib.layers.variance_scaling_initializer() )    
    biases3 = tf.Variable(0.5, dtype = tf.float32, name = "biases3")

    normalized = tf.nn.batch_normalization(inPH,train_mean, train_stddev,0.0, 1.0, 0.0, name = "normalized")
    tf.summary.scalar( 'normalized', tf.reduce_mean(normalized) )
    
    h1 = tf.nn.relu(tf.add(tf.matmul(normalized, weights1), biases1), name = "h1")
    tf.summary.scalar( 'mean1', tf.reduce_mean(h1) )
    tf.summary.histogram('histogram1', h1)    
    
    
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, weights2), biases2), name = "h2")
    tf.summary.scalar('mean2', tf.reduce_mean(h2) )    
    tf.summary.histogram('histogram2', h2)
        
    predOuts = tf.nn.softmax(tf.add(tf.matmul(h2, weights3), biases3), name = "predOuts")
    tf.summary.scalar('mean_out', tf.reduce_mean(predOuts) )
    tf.summary.scalar('max_out', tf.reduce_max(predOuts) )
    tf.summary.scalar('min_out', tf.reduce_min(predOuts) )    
    tf.summary.histogram('histogram_out', predOuts)
    
    learnRate = 0.01


    loss = tf.reduce_mean(tf.square(desOutPH-predOuts))
    tf.summary.scalar('mean_loss', tf.reduce_mean(loss) )    
    tf.summary.histogram('histogram_loss', loss)    
    
    optimizer = tf.train.AdamOptimizer(learnRate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(desOutPH, 1), tf.argmax(predOuts, 1))
    
    accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))
    tf.summary.scalar('mean_acc', tf.reduce_mean(accuracy) )    
    tf.summary.histogram('histogram_acc', accuracy)	
    
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter( folder_name + '/test' )    
    train_writer = tf.summary.FileWriter( folder_name + '/train' )
    nrEpochs =  int(len(train_data)/ int(100))
    print(nrEpochs)
    print(nrEpochs*3)
    nrRepetitions = 4
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer() )
        
        for x in range(int(nrRepetitions)):
            for i in range(int(nrEpochs)):
                if i % nrEpochs - 1 == 0 :
                    data_to_train = train_data[i*100:]
                    des_for_train  = des_out[i*100:]    
                else:        
                    data_to_train = train_data[i*100:(i+1)*100]                
                    des_for_train = des_out[i*100:(i+1)*100]

                trainData = {inPH : data_to_train, desOutPH : des_for_train, keep_prob : 0.15}    
                opt,currentLoss = sess.run([optimizer, loss], feed_dict = trainData)
                eval = sess.run (accuracy, feed_dict = trainData)
                
                summary = sess.run(merged, feed_dict = trainData)
                train_writer.add_summary(summary, x*nrEpochs + i)
                
                if i % 5 == 0:
                    testData = {inPH : test_data, desOutPH : des_out_test, keep_prob : 0.15}
                    
                    
                    output = sess.run (predOuts, feed_dict = testData)
                    eval = sess.run (accuracy, feed_dict = testData)
                    
                    summary = sess.run(merged, feed_dict = testData)            
                    
                    test_writer.add_summary(summary, x*nrEpochs +i)
                    print('Epoch = ', x*nrEpochs + i)
                    print('Loss = ', currentLoss) 
                    print('Accuracy = ', eval)
                    print('Output = ', output)
                    print('\n\n')
                if x == nrRepetitions - 1 and i == nrEpochs - 1:
                    saver.save(sess, "./model.ckpt")

                        
    #chkp.print_tensors_in_checkpoint_file("./model.ckpt", tensor_name='', all_tensors=True)                
    
def convert_to_pb():
    meta_graph = tf.train.import_meta_graph('./model.ckpt.meta')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    
    meta_graph.restore(sess, './checkpoint')
    output_node_names = "predOuts"
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))
    
    output_graph = "./model.pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    sess.close()

def load_model():
    sess = tf.Session()
    graph = tf.train.import_meta_graph('model.ckpt.meta')
    graph.restore(sess, tf.train.latest_checkpoint('./'))
    return graph

def predict_symbol(saver, key_points):
    y_pred = graph.get_tensor_by_name("predOuts")
    input = graph.get_tensor_by_name("inPH")
    sess = tf.Session(graph = graph)
    input_feed = { input: key_points }
    result = sess.run(y_pred , feed_dict = input_feed)
    print(result)
    #return result
    
# with open("train_data.csv") as csv_file:
    # csv_reader = csv.reader(csv_file, delimiter = ',')
    # data = list(csv_reader)
# for i in range(len(data)):
    # del data[i][-1]
    # if len(data[i]) != 44:
        # print(i)
        # print(len(data[i]))

    
#train_images, test_images = get_images(images_path)
#define_and_run_conv_net(train_images, test_images)
 
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  
                
# read_from_csv("train_data.csv", train_data, des_out)
# print("read from train")
# read_from_csv("test_data.csv", test_data, des_out_test)
# print("read from test")
# train_mean = np.mean(train_data)
# train_std = np.std(train_data)
# print(train_mean)
# print(train_std)
# define_and_run_neural_net(train_mean,train_std)
