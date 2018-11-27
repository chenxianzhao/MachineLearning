
#-*- coding:utf-8 -*-

#单隐藏层感知机
import numpy as np;
import math;


def Sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x));
sigmoid = np.vectorize(Sigmoid);
def partdataset(x):
    ret1 = [];
    ret2 = [];
    for d in x:
        ret1.append(d[:8]);
        ret2.append(d[8:]);
    return ret1,ret2;
class perceptron:

    input_size = 8;
    output_size = 1;
    hidden_size = 10;
    learnrate = 0.9;
    ac = 10
    def P(x):
        print(123);
    def __init__(self):

        self.i2h_Mat = np.random.random((self.input_size,self.hidden_size));
        self.h2o_Mat = np.random.random((self.hidden_size,self.output_size));
        self.hiddenbase = np.random.random(self.hidden_size);
        self.outputbase = np.random.random(self.output_size);
        #print(self.hiddenbase.shape);

    def cal(self,x):
        #print ("input ",x.shape,x);
        hiddeninput = np.dot(x,self.i2h_Mat);
        #print ("hidden in ",hiddeninput.shape,hiddeninput);
        self.hiddenoutput = sigmoid(hiddeninput+self.hiddenbase);
        #print ("hdiien out ",self.hiddenoutput.shape,self.hiddenoutput);
        outputI = np.dot(self.hiddenoutput,self.h2o_Mat);
        #print("outpuI ",outputI.shape,outputI);
        self.outputO = sigmoid(outputI+self.outputbase);

    def calg(self):
        self.g = self.outputO*(1-self.outputO)*(self.Eoutput-self.outputO);

    def getwhjgj(self,h):
        ret =0;
        for j in range(self.output_size):
            ret = ret+self.h2o_Mat[h][j]*self.g[j];
        return ret
    def cale(self):
        tmp1 = self.hiddenoutput*(1-self.hiddenoutput);
        for h in range(self.hidden_size):
            tmp1[h] = tmp1[h]*self.getwhjgj(h);
        self.e =  tmp1;
    def funcdeltah2o_Mat(self,i,j):
        self.ac = 11;
        return i*j;
    def getdeltaomega(self):
        self.deltaomega = np.zeros((self.hidden_size,self.output_size));
        for h in range(self.hidden_size):
            for j in range(self.output_size):
                self.deltaomega[h][j] = self.g[j]*self.hiddenoutput[h];
        self.deltaomega = self.deltaomega*self.learnrate;
    def getdeltatheta(self):
        self.deltatheta = self.g*(-self.learnrate);

    def getdeltanu(self):
        self.deltanu = np.zeros((self.input_size,self.hidden_size));
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                self.deltanu[i][h] = self.e[h]*self.input[i];
        self.deltanu = self.deltanu*self.learnrate;
    def getdeltagamma(self):
        self.deltagamma = self.e*(-self.learnrate);

    def update(self,x):
        self.Eoutput = np.array(x[8:]);
        #print ("exception output",self.Eoutput);
        self.input = np.array(x[:8]);
        #print ("input",self.input);
        self.cal(self.input);
        self.calg();
        self.cale();
        self.getdeltaomega();
        self.getdeltatheta();
        self.getdeltanu();
        self.getdeltagamma();

        self.h2o_Mat +=self.deltaomega;
        self.outputbase +=self.deltatheta;
        self.i2h_Mat +=self.deltanu;
        self.hiddenbase +=self.deltagamma;
    def valuation(self,dataset):
        ds,cl = partdataset(dataset);
        res = np.array([]);
        for data in ds:
            self.cal(np.array(data));
            res = np.append(res,self.outputO);
        print("res",res);

        sq = np.square(res-np.array(cl));
        np.set_printoptions(precision=3,suppress=True);
        print("msq:",np.mean(sq));
dataSet = [
        # 1
        [1, 2, 1, 0, 2, 1, 0.697, 0.460, 1],
        # 2
        [2, 2, 0, 0, 2, 1, 0.774, 0.376, 1],
        # 3
        [2, 2, 1, 0, 2, 1, 0.634, 0.264, 1],
        # 4
        [1, 2, 0, 0, 2, 1, 0.608, 0.318, 1],
        # 5
        [0, 2, 1, 0, 2, 1, 0.556, 0.215, 1],
        # 6
        [1, 1, 1, 0, 1, 0, 0.403, 0.237, 1],
        # 7
        [2, 1, 1, 1, 1, 0, 0.481, 0.149, 1],
        # 8
        [2, 1, 1, 0, 1, 1, 0.437, 0.211, 1],

        # ----------------------------------------------------
        # 9
        [2, 1, 0, 1, 1, 1, 0.666, 0.091, 0],
        # 10
        [1, 0, 2, 0, 0, 0, 0.243, 0.267, 0],
        # 11
        [0, 0, 2, 2, 0, 1, 0.245, 0.057, 0],
        # 12
        [0, 2, 1, 2, 0, 0, 0.343, 0.099, 0],
        # 13
        [1, 1, 1, 1, 2, 1, 0.639, 0.161, 0],
        # 14
        [0, 1, 0, 1, 2, 1, 0.657, 0.198, 0],
        # 15
        [2, 1, 1, 0, 1, 0, 0.360, 0.370, 0],
        # 16
        [0, 2, 1, 2, 0, 1, 0.593, 0.042, 0],
        # 17
        [1, 2, 0, 1, 1, 1, 0.719, 0.103, 0]
    ]

# 好瓜是1 坏瓜是0;

#zz = map(lambda x:x*2,dd)
#print(perceptron().cal(np.array([1,2,3,4,5,6,7,8])));
myperceptron = perceptron();
for round in range(500):
    for data in dataSet:
        # print(data);
        myperceptron.update(data);
myperceptron.valuation(dataSet);
