import csv
import pickle
import numpy as np
from main import *
import nn

pathToTestCase = "testcases/testcase_01.pkl"
load_test_case = pickle.load(open(pathToTestCase, 'rb'))

task_detail = {
	1: [4, 'Forward Pass'],
	2: [6, 'Forward + Backward Pass'],
	3: [2, 'Update weights'],
	4: [1, 'Check Relu'],
	5: [1, 'Check Relu Gradient'],
	6: [3, 'Check Softmax'],
	7: [7, 'Check Softmax Gradient'],
	8: [3, 'Check Cross Entropy Loss'],
	9: [3, 'Check Cross Entropy Loss Gradient']
}

def check_forward(task_number):
	print('='*20 + ' TASK '+str(task_number)+' '+str(task_detail[task_number][1])+' '+ '='*20)
	input_X = np.asarray(load_test_case['forward_input'])

	nn1 = nn.NeuralNetwork(0.0, 1, 1)
	nn1.addLayer(nn.FullyConnectedLayer(2,1,'relu'))
	nn1.addLayer(nn.FullyConnectedLayer(1,2,'softmax'))

	output_X = input_X

	ind = 0
	weights = load_test_case['forward_weights'] #[[0.16411168 0.2137831 ][0.03711599 0.02328956]]
	biases = load_test_case['forward_biases'] #[[2 3]]
	layers = nn1.layers
	for l in layers:
		l.weights = weights[ind]
		l.biases = biases[ind]
		ind+=1

	for l in nn1.layers:
		output_X = l.forwardpass(output_X)

	studentAnswer = output_X
	teacherAnswer = load_test_case['forward_output']

	teacherAnswer = np.round(teacherAnswer, 5)
	studentAnswer = np.round(studentAnswer, 5)

	print('Student Answer', studentAnswer)
	print('Correct Answer', teacherAnswer)

	print('Correct', np.array_equal(studentAnswer, teacherAnswer))
	return np.array_equal(studentAnswer, teacherAnswer)


def check_backward(task_number):
	print('='*20 + ' TASK '+str(task_number)+' '+str(task_detail[task_number][1])+' '+ '='*20)
	input_X = np.asarray(load_test_case['backward_input'])
	input_delta = np.asarray(load_test_case['backward_input_delta'])

	nn1 = nn.NeuralNetwork(0.0, .1, 1)
	nn1.addLayer(nn.FullyConnectedLayer(2,5,'relu'))
	nn1.addLayer(nn.FullyConnectedLayer(5,2,'softmax'))

	ind = 0
	weights = load_test_case['backward_weights']
	biases = load_test_case['backward_biases']
	layers = nn1.layers
	for l in layers:
		l.weights = weights[ind]
		l.biases = biases[ind]
		ind+=1

	activations = [input_X]
	for l in layers:
		activations.append(l.forwardpass(activations[-1]))

	# activations = load_test_case['backward_input_activations']

	weightsGrad = list()
	biasesGrad = list()
	delta = input_delta
	for i in range(len(layers)-1, -1, -1):
		delta = layers[i].backwardpass(activations[i], delta)
		weightsGrad.append(layers[i].weightsGrad)
		biasesGrad.append(layers[i].biasesGrad)

	studentAnswerdelta = delta
	studentAnswerweightsGrad = weightsGrad
	studentAnswerbiasesGrad = biasesGrad

	teacherAnswerdelta = load_test_case['backward_output']
	teacherAnswerweightsGrad = load_test_case['backward_weightsGrad']
	teacherAnswerbiasesGrad = load_test_case['backward_biasesGrad']

	teacherAnswerdelta = np.asarray(teacherAnswerdelta)
	teacherAnswerweightsGrad = np.asarray(teacherAnswerweightsGrad)
	teacherAnswerbiasesGrad = np.asarray(teacherAnswerbiasesGrad)

	studentAnswerdelta = np.round(studentAnswerdelta, 6)
	teacherAnswerdelta = np.round(teacherAnswerdelta, 6)
	studentAnswerweightsGrad = [np.round(x, 6) for x in studentAnswerweightsGrad]
	teacherAnswerweightsGrad = [np.round(x, 6) for x in teacherAnswerweightsGrad]
	studentAnswerbiasesGrad = [np.round(x, 6) for x in studentAnswerbiasesGrad]
	teacherAnswerbiasesGrad = [np.round(x, 6) for x in teacherAnswerbiasesGrad]

	print('Student Answer deltas', studentAnswerdelta)
	print('Correct Answer deltas', teacherAnswerdelta)
	print('Student Answer weights Gradient', studentAnswerweightsGrad)
	print('Correct Answer weights Gradient', teacherAnswerweightsGrad)
	print('Student Answer biases Gradient', studentAnswerbiasesGrad)
	print('Correct Answer biases Gradient', teacherAnswerbiasesGrad)


	print('Correct', np.array_equal(studentAnswerdelta, teacherAnswerdelta) and 
				np.all([np.array_equal(x, y) for x, y in zip(studentAnswerweightsGrad, teacherAnswerweightsGrad)]) and
				np.all([np.array_equal(x, y) for x, y in zip(studentAnswerbiasesGrad, teacherAnswerbiasesGrad)]))
	return (np.array_equal(studentAnswerdelta, teacherAnswerdelta) and 
			np.all([np.array_equal(x, y) for x, y in zip(studentAnswerweightsGrad, teacherAnswerweightsGrad)]) and
				np.all([np.array_equal(x, y) for x, y in zip(studentAnswerbiasesGrad, teacherAnswerbiasesGrad)]))

def check_updateweights(task_number):
	print('='*20 + ' TASK '+str(task_number)+' '+str(task_detail[task_number][1])+' '+ '='*20)
	nn1 = nn.NeuralNetwork(0.0, 1, 1)
	nn1.addLayer(nn.FullyConnectedLayer(2,1,'relu'))

	weights = load_test_case['updateweights_weights']
	biases = load_test_case['updateweights_biases']
	weightsGrad = load_test_case['updateweights_weightsGrad']
	biasesGrad = load_test_case['updateweights_biasesGrad']

	layer = nn1.layers[0]
	layer.weights = weights
	layer.biases = biases
	layer.weightsGrad = weightsGrad
	layer.biasesGrad = biasesGrad

	layer.updateWeights(0.01)

	studentAnswer = [layer.weights, layer.biases]
	teacherAnswer = load_test_case['updateweights_output']

	print('Student Answer', studentAnswer)
	print('Correct Answer', teacherAnswer)

	teacherAnswer_weight = np.round(teacherAnswer[0], 6)
	studentAnswer_weight = np.round(studentAnswer[0], 6)
	teacherAnswer_bias = np.round(teacherAnswer[1], 6)
	studentAnswer_bias = np.round(studentAnswer[1], 6)

	print('Correct', np.array_equal(studentAnswer[0], teacherAnswer[0]) and np.array_equal(studentAnswer[1], teacherAnswer[1]))
	return np.array_equal(studentAnswer[0], teacherAnswer[0]) and np.array_equal(studentAnswer[1], teacherAnswer[1])


def check_relu(task_number):
	print('='*20 + ' TASK '+str(task_number)+' '+str(task_detail[task_number][1])+' '+ '='*20)
	input_X = np.asarray(load_test_case['relu_input']).reshape(1,4)

	nn1 = nn.NeuralNetwork(0.0, 1, 1)
	nn1.addLayer(nn.FullyConnectedLayer(4,4,'relu'))

	output_X = input_X

	output_X = nn1.layers[0].relu_of_X(output_X)

	studentAnswer = output_X
	teacherAnswer = load_test_case['relu_output']

	teacherAnswer = np.round(teacherAnswer, 6)
	studentAnswer = np.round(studentAnswer, 6)

	print('Student Answer', studentAnswer)
	print('Correct Answer', teacherAnswer)
	print('Correct', np.array_equal(studentAnswer, teacherAnswer))
	return np.array_equal(studentAnswer, teacherAnswer)

def check_gardient_relu(task_number):
	print('='*20 + ' TASK '+str(task_number)+' '+str(task_detail[task_number][1])+' '+ '='*20)
	input_X = np.asarray(load_test_case['gardient_relu_input']).reshape(1,4)
	input_delta = np.asarray(load_test_case['gardient_relu_input_delta']).reshape(1,4)

	nn1 = nn.NeuralNetwork(0.0, 1, 1)
	nn1.addLayer(nn.FullyConnectedLayer(4,4,'relu'))

	output_X = input_X

	output_X = nn1.layers[0].gradient_relu_of_X(output_X, input_delta)

	studentAnswer = output_X
	teacherAnswer = load_test_case['gardient_relu_output']

	teacherAnswer = np.round(teacherAnswer, 6)
	studentAnswer = np.round(studentAnswer, 6)

	print('Student Answer', studentAnswer)
	print('Correct Answer', teacherAnswer)
	print('Correct', np.array_equal(studentAnswer, teacherAnswer))
	return np.array_equal(studentAnswer, teacherAnswer)


def check_softmax(task_number):
	print('='*20 + ' TASK '+str(task_number)+' '+str(task_detail[task_number][1])+' '+ '='*20)
	input_X = np.asarray(load_test_case['softmax_input']).reshape(1,4)

	nn1 = nn.NeuralNetwork(0.0, 1, 1)
	nn1.addLayer(nn.FullyConnectedLayer(4,4,'softmax'))

	output_X = input_X

	output_X = nn1.layers[0].softmax_of_X(output_X)

	studentAnswer = output_X

	teacherAnswer = load_test_case['softmax_output']

	teacherAnswer = np.round(teacherAnswer, 6)
	studentAnswer = np.round(studentAnswer, 6)

	print('Student Answer', studentAnswer)
	print('Correct Answer', teacherAnswer)
	print('Correct', np.array_equal(studentAnswer, teacherAnswer))
	return np.array_equal(studentAnswer, teacherAnswer)

def check_gardient_softmax(task_number):
	print('='*20 + ' TASK '+str(task_number)+' '+str(task_detail[task_number][1])+' '+ '='*20)
	input_X = np.asarray(load_test_case['gardient_softmax_input']).reshape(1,4)
	input_delta = np.asarray(load_test_case['gardient_softmax_input_delta']).reshape(1,4)

	nn1 = nn.NeuralNetwork(0.0, 1, 1)
	nn1.addLayer(nn.FullyConnectedLayer(4,4,'softmax'))

	output_X = input_X

	output_X = nn1.layers[0].gradient_softmax_of_X(output_X, input_delta)
	
	studentAnswer = output_X
	teacherAnswer = load_test_case['gardient_softmax_output']

	teacherAnswer = np.round(teacherAnswer, 6)
	studentAnswer = np.round(studentAnswer, 6)

	print('Student Answer', studentAnswer)
	print('Correct Answer', teacherAnswer)
	print('Correct', np.array_equal(studentAnswer, teacherAnswer))
	return np.array_equal(studentAnswer, teacherAnswer)

def check_crossEntropyLoss(task_number):
	print('='*20 + ' TASK '+str(task_number)+' '+str(task_detail[task_number][1])+' '+ '='*20)
	input_Y = np.asarray(load_test_case['crossEntropyLoss_input_Y']).reshape(2, 10)
	input_Y_pred = np.asarray(load_test_case['crossEntropyLoss_input_Y_pred']).reshape(2, 10)

	nn1 = nn.NeuralNetwork(0.0, 4, 1)

	output_Y = nn1.crossEntropyLoss(input_Y, input_Y_pred)
	studentAnswer = output_Y
	teacherAnswer = load_test_case['crossEntropyLoss_output']

	teacherAnswer = np.round(teacherAnswer, 6)
	studentAnswer = np.round(studentAnswer, 6)

	print('Student Answer', studentAnswer)
	print('Correct Answer', teacherAnswer)
	print('Correct', np.array_equal(studentAnswer, teacherAnswer))
	return np.array_equal(studentAnswer, teacherAnswer)

def check_crossEntropyDelta(task_number):
	print('='*20 + ' TASK '+str(task_number)+' '+str(task_detail[task_number][1])+' '+ '='*20)
	input_Y = np.asarray(load_test_case['crossEntropyDelta_input_Y']).reshape(2, 10)
	input_Y_pred = np.asarray(load_test_case['crossEntropyDelta_input_Y_pred']).reshape(2, 10)

	nn1 = nn.NeuralNetwork(0.0, 4, 1)

	output_Y = nn1.crossEntropyDelta(input_Y, input_Y_pred)
	studentAnswer = output_Y
	teacherAnswer = load_test_case['crossEntropyDelta_output']

	teacherAnswer = np.round(teacherAnswer, 6)
	studentAnswer = np.round(studentAnswer, 6)

	print('Student Answer', studentAnswer)
	print('Correct Answer', teacherAnswer)
	print('Correct', np.array_equal(studentAnswer, teacherAnswer))
	return np.array_equal(studentAnswer, teacherAnswer)


if __name__ == "__main__":

	np.random.seed(42)
	print()
	correct_status = False
	total_marks = 0

	try:
		correct_status = check_forward(1)
		total_marks+=(correct_status*task_detail[1][0])
	except Exception as e:
		print("Error "+str(e)+" occured in task ", 1)
		print("Correct False")

	try:
		correct_status = check_backward(2)
		total_marks+=(correct_status*task_detail[2][0])
	except Exception as e:
		print("Error "+str(e)+" occured in task ", 2)
		print("Correct False")

	try:
		correct_status = check_updateweights(3)
		total_marks+=(correct_status*task_detail[3][0])
	except Exception as e:
		print("Error "+str(e)+" occured in task ", 3)
		print("Correct False")

	try:
		correct_status = check_relu(4)
		total_marks+=(correct_status*task_detail[4][0])
	except Exception as e:
		print("Error "+str(e)+" occured in task ", 4)
		print("Correct False")

	try:
		correct_status = check_gardient_relu(5)
		total_marks+=(correct_status*task_detail[5][0])
	except Exception as e:
		print("Error "+str(e)+" occured in task ", 5)
		print("Correct False")

	try:
		correct_status = check_softmax(6)
		total_marks+=(correct_status*task_detail[6][0])
	except Exception as e:
		print("Error "+str(e)+" occured in task ", 6)
		print("Correct False")

	try:
		correct_status = check_gardient_softmax(7)
		total_marks+=(correct_status*task_detail[7][0])
	except Exception as e:
		print("Error "+str(e)+" occured in task ", 7)
		print("Correct False")

	try:
		correct_status = check_crossEntropyLoss(8)
		total_marks+=(correct_status*task_detail[8][0])
	except Exception as e:
		print("Error "+str(e)+" occured in task ", 8)
		print("Correct False")

	try:
		correct_status = check_crossEntropyDelta(9)
		total_marks+=(correct_status*task_detail[9][0])
	except Exception as e:
		print("Error "+str(e)+" occured in task ", 9)
		print("Correct False")

	print('='*20 + ' TASK Finish ' + '='*20)

	full_marks = 0
	for x in range(len(task_detail)):
		full_marks += task_detail[x+1][0]

	print('    You got', total_marks, 'Marks Out of', full_marks, 'for', pathToTestCase.split('/')[1].split('.')[0])
	print('='*53)
	print()
