import numpy as np
import random as rdm

np.random.seed(7)

def batch_maker(X,Y,batch_size):
    xn=[]
    yn=[]
    for i in range(batch_size):
        pos=rdm.randrange(0,len(X))
        xn.append(X[pos])
        yn.append(Y[pos])
    return xn,yn

class NeuralNetwork:
    
    def __init__(self, lr, batchSize, epochs):
        # Method to initialize a Neural Network Object
        # Parameters
        # lr - learning rate
        # batchSize - Mini batch size
        # epochs - Number of epochs for training
        self.lr = lr
        self.batchSize = batchSize
        self.epochs = epochs
        self.layers = []
    
    def addLayer(self, layer):
        # Method to add layers to the Neural Network
        self.layers.append(layer)
    
    def train(self, trainX, trainY, validX=None, validY=None):
        # Method for training the Neural Network
        # Input
        # trainX - A list of training input data to the neural network
        # trainY - Corresponding list of training data labels
        # validX - A list of validation input data to the neural network
        # validY - Corresponding list of validation data labels
        
        # The methods trains the weights and baises using the training data(trainX, trainY)
        # Feel free to print accuracy at different points using the validate() or computerAccuracy() functions of this class
        ###############################################
        # TASK 2c (Marks 0) - YOUR CODE HERE
        
        for i in range(self.epochs):
            x,y=batch_maker(trainX,trainY,self.batchSize)
            A=[]
            A.append(x)
            for j in self.layers:
                out=j.forwardpass(A[-1])
                A.append(out)
                
            deltta=self.crossEntropyDelta(y,A[-1])
            
            rev_layer=self.layers
            #rev_layer.reverse()
            
            
            for k in range(len(rev_layer)-1,-1,-1):
                deltta=self.layers[k].backwardpass(A[k],deltta)
        
            
            for l in self.layers:
                l.updateWeights(self.lr)
            
            #sumerror=sum(self.crossEntropyLoss(y,out))
            accuracy=self.computeAccuracy(y,out)
            #print('the sum of CEL is '+str(sumerror)+' and accuracy is '+(accuracy))
            print('the accuracy in epoch '+ str(i)+' :  '+str(accuracy))
            #if(accuracy==100.0):
                #break
                
        if validX is not None:
            if validY is not None:
                pred,acc=self.validate(validX,validY)
                print('validation accuracy is ',acc)
        ###############################################
        
    def crossEntropyLoss(self, Y, predictions):
        # Input 
        # Y : Ground truth labels (encoded as 1-hot vectors) | shape = batchSize x number of output labels
        # predictions : Predictions of the model | shape = batchSize x number of output labels
        # Returns the cross-entropy loss between the predictions and the ground truth labels | shape = scalar
        ###############################################
        # TASK 2a (Marks 3) - YOUR CODE HERE
        error=0
        for i in range(len(Y)):
            error-=np.matmul(Y[i],np.log(predictions[i]))
        return error
        ###############################################

    def crossEntropyDelta(self, Y, predictions):
        # Input 
        # Y : Ground truth labels (encoded as 1-hot vectors) | shape = batchSize x number of output labels
        # predictions : Predictions of the model | shape = batchSize x number of output labels
        # Returns the derivative of the loss with respect to the last layer outputs, ie dL/dp_i where p_i is the ith 
        #        output of the last layer of the network | shape = batchSize x number of output labels
        ###############################################
        # TASK 2b (Marks 3) - YOUR CODE HERE
        grad=np.array(Y).astype(float)
        for i in range(len(Y)):
            for j in range(len(Y[0])):
                grad[i][j]= (-Y[i][j])/(predictions[i][j]+0.0000000000001)
        return grad
        ###############################################
        
    def computeAccuracy(self, Y, predictions):
        # Returns the accuracy given the true labels Y and final output of the model
        correct = 0
        for i in range(len(Y)):
            if np.argmax(Y[i]) == np.argmax(predictions[i]):
                correct += 1
        accuracy = (float(correct) / len(Y)) * 100
        return accuracy

    def validate(self, validX, validY):
        # Input 
        # validX : Validation Input Data
        # validY : Validation Labels
        # Returns the predictions and validation accuracy evaluated over the current neural network model
        valActivations = self.predict(validX)
        pred = np.argmax(valActivations, axis=1)
        if validY is not None:
            valAcc = self.computeAccuracy(validY, valActivations)
            return pred, valAcc
        else:
            return pred, None

    def predict(self, X):
        # Input
        # X : Current Batch of Input Data as an nparray
        # Output
        # Returns the predictions made by the model (which are the activations output by the last layer)
        # Note: Activations at the first layer(input layer) is X itself        
        activations = X
        for l in self.layers:
            activations = l.forwardpass(activations)
        return activations







class FullyConnectedLayer:
    def __init__(self, in_nodes, out_nodes, activation):
        # Method to initialize a Fully Connected Layer
        # Parameters
        # in_nodes - number of input nodes of this layer
        # out_nodes - number of output nodes of this layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.activation = activation
        # Stores a quantity that is computed in the forward pass but actually used in the backward pass. Try to identify
        # this quantity to avoid recomputing it in the backward pass and hence, speed up computation
        self.data = None

        # Create np arrays of appropriate sizes for weights and biases and initialise them as you see fit
        ###############################################
        # TASK 1a (Marks 0) - YOUR CODE HERE
        self.weights = np.random.rand(self.in_nodes,self.out_nodes)
        #self.weights = np.ones((self.in_nodes,self.out_nodes))
        self.biases = np.random.rand(self.out_nodes)
        
        #self.biases = np.ones(self.out_nodes)
        
        ###############################################
        # NOTE: You must NOT change the above code but you can add extra variables if necessary
        
        # Store the gradients with respect to the weights and biases in these variables during the backward pass
        self.weightsGrad = np.random.rand(self.in_nodes,self.out_nodes)
        #self.weightsGrad = np.ones((self.in_nodes,self.out_nodes))
        self.biasesGrad = np.random.rand(self.out_nodes)
        #self.biasesGrad = np.ones(self.out_nodes)
        

    def relu_of_X(self, X):
        # Input
        # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
        # Returns: Activations after one forward pass through this relu layer | shape: batchSize x self.out_nodes
        # This will only be called for layers with activation relu
        ###############################################
        # TASK 1b (Marks 1) - YOUR CODE HERE
        X=np.array(X).astype(float)
        for i in range(len(X)):
            for j in range(len(X[0])):
                X[i][j]=max(0,X[i][j])
        
        return X
        ###############################################

    def gradient_relu_of_X(self, X, delta):
        # Input
        # data : Output from next layer/input | shape: batchSize x self.out_nodes
        # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
        # Returns: Current del_Error to pass to current layer in backward pass through relu layer | shape: batchSize x self.out_nodes
        # This will only be called for layers with activation relu amd during backwardpass
        ###############################################
        # TASK 1e (Marks 1) - YOUR CODE HERE
        new_delta=np.array(delta).astype(float)
        for i in range(len(delta)):
            for j in range(len(delta[0])):
                if(X[i][j]<=0):
                    new_delta[i][j]=0*delta[i][j]
                else:
                    new_delta[i][j]=1*delta[i][j]
                
        return new_delta
        ###############################################

    def softmax_of_X(self, X):
        # Input
        # data : Output from current layer/input for Activation | shape: batchSize x self.out_nodes
        # Returns: Activations after one forward pass through this softmax layer | shape: batchSize x self.out_nodes
        # This will only be called for layers with activation softmax
        ###############################################
        # TASK 1c (Marks 3) - YOUR CODE HERE
        X=np.exp(X)
        for i in range(len(X)):
            sumor=np.sum(X[i])
            #for j in range(len(X[0])):
            #    sumor+=X[i][j]
            X[i]=np.true_divide(X[i],sumor)
        
        return X
        ###############################################

    def gradient_softmax_of_X(self, X, delta):
        # Input
        # data : Output from next layer/input | shape: batchSize x self.out_nodes
        # delta : del_Error/ del_activation_curr | shape: batchSize x self.out_nodes
        # Returns: Current del_Error to pass to current layer in backward pass through softmax layer | shape: batchSize x self.out_nodes
        # This will only be called for layers with activation softmax amd during backwardpass
        # Hint: You might need to compute Jacobian first
        ###############################################
        # TASK 1f (Marks 7) - YOUR CODE HERE
        grad=np.array(delta).astype(float)
        for i in range(len(X)):
            DS=np.zeros((len(X[0]),len(X[0])))
            for j in range(len(X[0])):
                for k in range(len(X[0])):
                    if(j==k):
                        DS[j][k]=X[i][j]*(1-X[i][k])
                    else:
                        DS[j][k]=-X[i][j]*X[i][k]
                        
            grad[i]=np.matmul(np.array(delta[i]),DS.transpose())
        return grad
        ###############################################

    def forwardpass(self, X):
        # Input
        # activations : Activations from previous layer/input | shape: batchSize x self.in_nodes
        # Returns: Activations after one forward pass through this layer | shape: batchSize x self.out_nodes
        # You may need to write different code for different activation layers
        ###############################################
        # TASK 1d (Marks 4) - YOUR CODE HERE
        #print('forward pass with activation '+self.activation+'in nodes and out nodes are'+str(self.in_nodes)+'X'+str(self.out_nodes))
        Z=np.dot(np.array(X).astype(float),np.array(self.weights).astype(float))+np.array(self.biases).astype(float)
        
        if self.activation == 'relu':
            A=self.relu_of_X(Z)
            self.data=A
            return A
        elif self.activation == 'softmax':
            A=self.softmax_of_X(Z)
            self.data=A
            return A
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        ###############################################

    def backwardpass(self, activation_prev, delta):
        # Input
        # activation_prev : Output from next layer/input | shape: batchSize x self.out_nodes]
        # delta : del_Error/ del_activation_curr | shape: self.out_nodes
        # Output
        # new_delta : del_Error/ del_activation_prev | shape: self.in_nodes
        # You may need to write different code for different activation layers

        # Just compute and store the gradients here - do not make the actual updates
        ###############################################
        # TASK 1g (Marks 6) - YOUR CODE HERE
        #print(self.activation)
        #print('backward pass with activation '+self.activation+'in nodes and out nodes are'+str(self.in_nodes)+'X'+str(self.out_nodes))
        if self.activation == 'relu':
            inp_delta = self.gradient_relu_of_X(self.data, delta)
        elif self.activation == 'softmax':
            inp_delta = self.gradient_softmax_of_X(self.data, delta)
        else:
            print("ERROR: Incorrect activation specified: " + self.activation)
            exit()
        
        #print(np.shape(inp_delta)) 
        #print(np.shape(self.weights))
        new_delta= inp_delta @ self.weights.T
        
        #print(new_delta)
        
        #print(activation_prev)
        self.weightsGrad=np.array(activation_prev).T @ inp_delta
        #self.weightsGrad=activation_prev.T @ inp_delta
        #print(np.shape(self.weightsGrad))
        biases=np.ones(inp_delta.shape[0])
        self.biasesGrad=biases.T @ inp_delta
        #self.biasesGrad=inp_delta        
        
        return new_delta
        ###############################################
        
    def updateWeights(self, lr):
        #print('updating wght with activation '+self.activation+'in nodes and out nodes are'+str(self.in_nodes)+'X'+str(self.out_nodes))
        #print(str(len(self.weights))+' '+str(len(self.weights[0])))
        #print(str(len(self.weightsGrad))+' '+str(len(self.weightsGrad[0])))
        
        #print(str(len(self.biases)))
        #print(str(len(self.biasesGrad)))
        # Input
        # lr: Learning rate being used
        # Output: None
        # This function should actually update the weights using the gradients computed in the backwardpass
        ###############################################
        # TASK 1h (Marks 2) - YOUR CODE HERE
       
        #print(self.weightsGrad)
        self.weights-=lr*self.weightsGrad
        self.biases-=lr*self.biasesGrad
        ###############################################
        