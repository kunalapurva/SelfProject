import numpy as np
import csv
import matplotlib.pyplot as plt




def plot_all_the_models(xtrainprocessed, ytrainprocessed, epoch, lr):
    dims = len(xtrainprocessed[0])
    Mmse = LinearRegressor(dims)
    Mrmse = LinearRegressor(dims)
    Mmae = LinearRegressor(dims)
    Mlogcosh = LinearRegressor(dims)
    Mmse.train(xtrainprocessed, ytrainprocessed,'mean_squared_loss','mean_squared_gradient' , epoch, lr)
    Mrmse.train(xtrainprocessed, ytrainprocessed,'mean_squared_loss','root_mean_squared_gradient' , epoch, lr)
    Mmae.train(xtrainprocessed, ytrainprocessed,'mean_squared_loss','mean_absolute_gradient' , epoch, lr)
    Mlogcosh.train(xtrainprocessed, ytrainprocessed,'mean_squared_loss','mean_log_cosh_gradient' , epoch, lr)
    
    xvalue = []
    for i in range(epoch):
            xvalue.append(i)
    
    plt.plot(xvalue, Mmse.error, color='g',label='mse')
    plt.plot(xvalue, Mrmse.error, color='r',label='rmse')
    plt.plot(xvalue, Mlogcosh.error, color='y',label='logcosh')
    plt.plot(xvalue, Mmae.error, color='b',label='mae')
    plt.savefig('cmp_e'+str(epoch)+'lr'+str(lr))
    
    
    
    
    

def mean_squared_loss(xdata, ydata, weights):
    y_predict = np.matmul(xdata, weights)
    error = ydata - y_predict
    sum_error = 0
    for i in range(len(error)):
        sum_error += pow(error[i], 2)
    sum_error = (sum_error) / len(xdata);
    return sum_error


def mean_squared_gradient(xdata, ydata, weights):
    y_predict = np.matmul(xdata, weights)
    error = ydata - y_predict
    gradient = -(1.0 / len(xdata)) *np.matmul(xdata.transpose(),error) #error.dot(xdata)
    return gradient


def mean_absolute_loss(xdata, ydata, weights):
    y_predict = np.matmul(xdata, weights)
    error = abs(ydata - y_predict)
    sum_error = sum(error)
    sum_error = sum_error / len(xdata);
    return sum_error


def mean_absolute_gradient(xdata, ydata, weights):
    y_predict = np.matmul(xdata, weights)
    error = ydata - y_predict
    error_sign = np.sign(error)
    gradient = -(1.0 / len(xdata)) * error_sign.dot(xdata)
    return gradient


def mean_log_cosh_loss(xdata, ydata, weights):
    y_predict = np.matmul(xdata, weights)
    err = abs(y_predict - ydata)
    for i in range(len(err)):
        if abs(err[i]) > 300:
            err[i] = 300
    error = np.log(np.cosh(err))
    sum_error = sum(error)
    sum_error = sum_error / len(xdata);
    return sum_error


def mean_log_cosh_gradient(xdata, ydata, weights):
    y_predict = np.matmul(xdata, weights)
    error = ydata - y_predict
    tanhp = np.tanh(error)
    gradient = -(1.0 / len(xdata)) * tanhp.dot(xdata)
    return gradient


def root_mean_squared_loss(xdata, ydata, weights):
    y_predict = np.matmul(xdata, weights)
    error = ydata - y_predict
    sum_error = 0
    for i in range(len(error)):
        sum_error += pow(error[i], 2)
    sum_error = (sum_error) / len(xdata);
    sum_error=np.sqrt(sum_error)
    return sum_error


def root_mean_squared_gradient(xdata, ydata, weights):
    y_predict = np.matmul(xdata, weights)
    error = ydata - y_predict
    errsqrt = 1.0 / np.sqrt(error.dot(error.transpose()))
    gradient = -(1.0 / np.sqrt(len(xdata))) * error.dot(xdata) * errsqrt
    return gradient


class LinearRegressor:
    weights = []
    error=[]

    def __init__(self, dims):
        self.weights = []
        for i in range(dims):
            self.weights.append(1)
      
        self.weights = np.array(self.weights)

    def train(self, xtrain, ytrain, loss_function, gradient_function, epoch=500, lr=0.250):
        self.error = []
        for i in range(epoch):

            gradient = globals()[gradient_function](xtrain, ytrain, self.weights)
            self.weights = self.weights - lr * gradient
            self.error.append(globals()[loss_function](xtrain, ytrain, self.weights))
            

def predict(self, xtest):
    alku = []
    alku.append(['instance (id)', 'count'])
    # This returns your prediction on xtest
    y_predicted = np.matmul(xtest, self.weights)
    for i in range(0, len(y_predicted)):
        if (y_predicted[i] < 0):
            y_predicted[i] = 12
        else:
            y_predicted[i] = round(y_predicted[i])
    for i in range(0, len(y_predicted)):
        alku.append([i, y_predicted[i]])

    with open('Prediction.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(alku)
        
    return y_predicted

def ploterror(self):
    year = []
    for i in range(max(len(self.error[1]),len(self.error[0]))):
        year.append(i)
            
    plt.plot(year, self.error[0], color='g',label='logcosh')
        



def read_dataset(trainfile, testfile):
    '''
	Reads the input data from train and test files and 
	Returns the matrices Xtrain : [N X D] and Ytrain : [N X 1] and Xtest : [M X D] 
	where D is number of features and N is the number of train rows and M is the number of test rows
	'''
    xtrain = []
    ytrain = []
    xtest = []

    with open(trainfile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            xtrain.append(row[:-1])
            ytrain.append(row[-1])

    with open(testfile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            xtest.append(row)

    return xtrain, ytrain, xtest


def preprocess_dataset(xdata, ydata=None):
    xpd = []
    xddata=[]
    xddata1=[]
    xdate=[]
   
    for i in range(0, len(xdata)):
        xdate.append(int(xdata[i][1][8:]))
        xdata[i][1] = int(xdata[i][1][5:7])
        
        if (xdata[i][5] == 'Monday'):
            xdata[i][5] = 1
        if (xdata[i][5] == 'Tuesday'):
            xdata[i][5] = 2
        if (xdata[i][5] == 'Wednesday'):
            xdata[i][5] = 3
        if (xdata[i][5] == 'Thursday'):
            xdata[i][5] = 4
        if (xdata[i][5] == 'Friday'):
            xdata[i][5] = 5
        if (xdata[i][5] == 'Saturday'):
            xdata[i][5] = 6
        if (xdata[i][5] == 'Sunday'):
            xdata[i][5] = 7
        xdata[i][10]=float(xdata[i][10])
        xdata[i][11]=float(xdata[i][11])
        xddata.append(xdata[i][10])
        xddata1.append(xdata[i][11])
            
    MM=min(xddata)
    MA=max(xddata)
    
    MM1=min(xddata1)
    MA1=max(xddata1)
    
    for i in range(0, len(xdata)):
        a=[]
        for j in range(87):
            a.append(0)
        a.append(1)
        a[int(xdata[i][2]) - 1] = 1
        a[4 + int(xdata[i][3])] = 1
        a[28] = int(xdata[i][4])
        a[28 + int(xdata[i][5])] = 1
        a[36] = int(xdata[i][6])
        a[36 + int(xdata[i][7])] = 1
        a[41] = xdata[i][8]
        a[42] =(xdata[i][10]-MM)/(MA-MM) #xdata[i][10] 
        a[43] =(xdata[i][11]-MM1)/(MA1-MM1) # xdata[i][11]
        a[44] = xdata[i][9]
        a[44+int(xdata[i][1])]=1
        a[56+int(xdate[i])]=1
        xpd.append(a)

    # print(xpd)

    for i in range(0, len(xpd)):
        for j in range(0, len(xpd[0])):
            xpd[i][j] = float(xpd[i][j])

    if (ydata == None):
        return np.array(xpd)
    else:
        for i in range(0, len(ydata)):
            ydata[i] = float(ydata[i])

        return np.array(xpd), np.array(ydata)


xtrain, ytrain, xtest = read_dataset('train.csv', 'test.csv')
xtrainprocessed, ytrainprocessed = preprocess_dataset(xtrain, ytrain)
epoch=300 #500
lr=0.4 #31
plot_all_the_models(xtrainprocessed, ytrainprocessed, epoch, lr)
print('mse:green,rmse:red,mae:blue,logcosh:yellow')
