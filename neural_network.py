import tensorflow as tf
import os
import numpy as np
import csv
import random
import time

train_data = list()
des_out = list()
test_data = list()
des_out_test = list()
folder_name = 'E:/test/log_at_{}'.format(int(time.time()))

def convert_to_prob(letter):
	if letter == ' A':
		return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	elif letter == ' B':
		return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	elif letter == ' C':
		return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	elif letter == ' D':
		return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
	elif letter == ' F':
		return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	elif letter == ' G':
		return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
	elif letter == ' I':
		return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
	elif letter == ' P':
		return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]		
	elif letter == ' V':
		return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
	elif letter == ' W':
		return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
	elif letter ==  ' Y':
		return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
		
def read_from_csv(file, input_list, label_list):

	with open(file) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		data = list(csv_reader)
	random.shuffle(data)
	random.shuffle(data)
	random.shuffle(data)
	random.shuffle(data)
	for row in data:
		label_list.append( convert_to_prob(row[len(row)-1]) )
		del row[-1]
		for x in range(len(row)):
			row[x] = float(row[x])
		input_list.append(row)
						
def define_and_run_neural_net(train_mean, train_stddev):
	global train_data
	global des_out
	global test_data
	global des_out_test
	

	inLayerSize = 20
	hiddenLayerSize1 = 40	
	hiddenLayerSize2 = 120
	outLayerSize = 11

	inPH = tf.placeholder(tf.float32, [None, inLayerSize] )
	desOutPH = tf.placeholder(tf.float32, [None, outLayerSize] )

	
	weights1 = tf.get_variable("weights1", [inLayerSize,hiddenLayerSize1], initializer=tf.contrib.layers.variance_scaling_initializer() )
	biases1 = tf.Variable(0.0, dtype = tf.float32)	
	weights2 = tf.get_variable("weights2", [hiddenLayerSize1, hiddenLayerSize2], initializer=tf.contrib.layers.variance_scaling_initializer() )
	biases2 = tf.Variable(0.0, dtype = tf.float32)
	weights3 = tf.get_variable("weights3", [hiddenLayerSize2, outLayerSize], initializer=tf.contrib.layers.variance_scaling_initializer() )	
	biases3 = tf.Variable(0.0, dtype = tf.float32)
	
	normalized = tf.nn.batch_normalization(inPH,train_mean, train_stddev,0.0, 1.0, 0.0)
	tf.summary.scalar( 'normalized', tf.reduce_mean(normalized) )
	h1 = tf.nn.relu(tf.add(tf.matmul(normalized, weights1), biases1))
	tf.summary.scalar( 'mean1', tf.reduce_mean(h1) )
	tf.summary.histogram('histogram1', h1)	
	
	h2 = tf.nn.relu(tf.add(tf.matmul(h1, weights2), biases2))
	tf.summary.scalar('mean2', tf.reduce_mean(h2) )	
	tf.summary.histogram('histogram2', h2)
	
	predOuts = tf.nn.softmax(tf.add(tf.matmul(h2, weights3), biases3))
	tf.summary.scalar('mean_out', tf.reduce_mean(predOuts) )
	tf.summary.scalar('max_out', tf.reduce_max(predOuts) )
	tf.summary.scalar('min_out', tf.reduce_min(predOuts) )	
	tf.summary.histogram('histogram_out', predOuts)
	
	learnRate = 0.01

	loss = tf.reduce_mean(tf.square(desOutPH - predOuts))
	tf.summary.scalar('mean_loss', tf.reduce_mean(loss) )	
	tf.summary.histogram('histogram_loss', loss)	
	
	optimizer = tf.train.AdamOptimizer(learnRate).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(desOutPH, 1), tf.argmax(predOuts, 1))
	
	accuracy = tf.reduce_mean (tf.cast (correct_prediction, tf.float32))
	tf.summary.scalar('mean_acc', tf.reduce_mean(accuracy) )	
	tf.summary.histogram('histogram_acc', accuracy)	
	

	merged = tf.summary.merge_all()
	test_writer = tf.summary.FileWriter( folder_name + '/test' )	
	train_writer = tf.summary.FileWriter( folder_name + '/train' )
	
	nrEpochs =  int(len(train_data)/ int(100))
	nrRepetitions = 100
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

				trainData = {inPH : train_data, desOutPH : des_out}	
				opt,currentLoss = sess.run([optimizer, loss], feed_dict = trainData)
				eval = sess.run (accuracy, feed_dict = trainData)
				
				summary = sess.run(merged, feed_dict = trainData)
				train_writer.add_summary(summary, x*100+i)
				
				if i % 5 == 0:
					testData = {inPH : test_data, desOutPH : des_out_test}
					
					
					output = sess.run (predOuts, feed_dict = testData)
					eval = sess.run (accuracy, feed_dict = testData)
					
					summary = sess.run(merged, feed_dict = testData)			
					
					test_writer.add_summary(summary, x*100+i)
					print('Epoch = ', x*100 + i)
					print('Loss = ', currentLoss) 
					print('Accuracy = ', eval)
					print('Output = ', output)
					print('\n\n')

		
	
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
				
read_from_csv("train_data.csv", train_data, des_out)
read_from_csv("test_data.csv", test_data, des_out_test)



#train_data = np.array(train_data).astype(np.float)
#des_out = np.array(des_out).astype(np.float)
#print(type(train_data[0][0]))
#print(type(des_out[0][0]))

# print(type(train_data[0][0]))
# print(type(des_out[0][0]))
# print(type(test_data[0][0]))
# print(type(des_out_test[0][0]))


train_mean = np.mean(train_data)
train_std = np.std(train_data)

define_and_run_neural_net(train_mean,train_std)
