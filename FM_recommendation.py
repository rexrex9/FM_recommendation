from mxnet import nd,autograd
from mxnet import gluon
import readData

epochs=2
batchSize=1000
features=42

def sigmoid(x):
    return 1/(1+nd.exp(-x))

class Net():
    def __init__(self):
        self.k=10
        self.b = nd.random_normal(shape=(1, 1),scale=0.01,dtype='float64')
        self.w = nd.random_normal(shape=(features, 1),scale=0.01,dtype='float64')
        self.bw = nd.random_normal(shape=(features,self.k),scale=0.0001,dtype='float64')
        self.params = [self.w, self.b,self.bw]

        for param in self.params:
            param.attach_grad()

    def __getTwoCross(self,X):
        i=0
        t=None
        for x in X:
            a=x.reshape(features,1)
            b=x.reshape(1,features)
            c=nd.dot(a,b)
            bbw=self.bw.reshape(self.k,features)
            ww = nd.dot(self.bw,bbw)

            s=nd.sum(ww*c).reshape((1,1))
            if i==0:
                t=nd.array(s)
            else:
                t=nd.concat(t,s,dim=0)
            i+=1
        return t

    def net(self,x):
        aa=self.__getTwoCross(x)
        a=self.b + nd.dot(x, self.w) + aa
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
