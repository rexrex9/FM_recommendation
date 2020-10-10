__author__='雷克斯掷骰子'

from mxnet import nd,autograd
from mxnet import gluon
import readData


epochs=10
batchSize=1000
features=42

def sigmoid(x):
    return 1/(1+nd.exp(-x))

class Net():
    def __init__(self):
        self.w0 = nd.random_normal(shape=(1,1),scale=0.01,dtype='float64')
        self.w = nd.random_normal(shape=(features, 1),scale=0.01,dtype='float64')
        self.params = [self.w, self.w0]

        for param in self.params:
            param.attach_grad()

    def net(self,x):
        a=nd.dot(x, self.w) + self.w0
        b=sigmoid(a)
        return b


    def SGD(self,lr):
        for param in self.params:
            param[:] = param - lr * param.grad

def dataIter(batch_size,trainX, trainY):
    dataset = gluon.data.ArrayDataset(trainX, trainY)
    train_data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
    return train_data_iter


def train( trainX, trainY):
    train_data_iter=dataIter(batchSize,trainX,trainY)
    lenTrainY=len(trainY)
    net = Net()
    lr=0.0001
    sigmoidBCEloss= gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    for e in range(epochs):
        total_loss = 0
        for x,y in train_data_iter:
            with autograd.record():
                y_hat=net.net(x)
                loss = sigmoidBCEloss(y_hat,y)
            loss.backward()
            net.SGD(lr)
            total_loss += nd.sum(loss).asscalar()
        print("Epoch %d, average loss:%f" % (e, total_loss / lenTrainY))
    return net



if __name__ == '__main__':
    trainX, trainY, testX, testY = readData.read_data()
    net=train(trainX,trainY)





