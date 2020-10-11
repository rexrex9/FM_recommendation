__author__='雷克斯掷骰子'

from mxnet import nd,autograd
from mxnet import gluon
import readData
from tqdm import tqdm

epochs=10
batchSize=1000
features=42

def sigmoid(x):
    return 1/(1+nd.exp(-x))

class Net():
    def __init__(self):
        self.w0 = nd.random_normal(shape=(1, 1),scale=0.01,dtype='float64')
        self.w = nd.random_normal(shape=(features, 1),scale=0.01,dtype='float64')
        self.bw = nd.random_normal(shape=(int((features*(features-1))/2),1),scale=0.0001,dtype='float64')
        self.params = [self.w, self.w0,self.bw]

        for param in self.params:
            param.attach_grad()

    def __getTwoCross(self,X):
        batch=0
        t=None
        for x in tqdm(X):
            i=0
            s=0
            for j1 in range(len(x)):
                for j2 in range(j1+1,len(x)):
                    s += (self.bw[i]*x[j1]*x[j2]).asscalar()
                    i += 1
            s=nd.array([[s]],dtype='float64')
            if batch==0:
                t=nd.array(s)
            else:
                t=nd.concat(t,s,dim=0)
            batch+=1
        return t

    def net(self,x):
        aa=self.__getTwoCross(x)
        a=self.w0 + nd.dot(x, self.w) + aa
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
        for x,y in tqdm(train_data_iter):
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
