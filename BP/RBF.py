#-*- coding:utf-8 -*-
import numpy as np;
dataset = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]];

#RBF
class RBF:
    input_size = 2;
    output_size = 1;
    rbf_size = 4;
    learnrate = 0.9;

    def __init__(self):
        print("RBF init");
        self.omega = np.random.random((self.rbf_size,self.output_size));
        self.c = np.array([[0,0],[0,1],[1,1],[1,0]]);
        self.beta = np.random.random(self.rbf_size);
    def norm2(self,i):
        tmp = self.input - self.c[i];
        tmpsqu = np.square(tmp);
        tmpsum = np.sum(tmpsqu);
        return tmpsum;

    def rbf(self,i):

        tmpsum = self.norm2(i);
        #print(tmpsum.__class__)
        return np.exp(-self.beta[i]*self.beta[i]*tmpsum);
    def cal(self,x):
        self.input =np.array(x);

        self.hiddenoutput = np.zeros(self.rbf_size);
        for i in range(self.rbf_size):
            self.hiddenoutput[i] = self.rbf(i);
        self.output = np.dot(self.hiddenoutput,self.omega);
        print("output" ,self.output);
    def update(self,x,o):
        self.cal(x);
        self.o = np.array(o);
        self.deltaomega = np.zeros((self.rbf_size,1));
        for i in range(self.rbf_size):
            #print(self.deltaomega[i]);
            self.deltaomega[i][0] = self.rbf(i);
        #print("before",(self.output-self.o).reshape(1,1));
        self.deltaomega = np.dot(self.deltaomega,(self.output-self.o).reshape(1,1));
        #print("after1", self.deltaomega);
        self.deltaomega = self.deltaomega*(-self.learnrate);
        #print("after2",self.deltaomega);
        #print(self.omega);
        self.deltabeta = np.dot((self.o-self.output).reshape(1,1),self.omega.T);
        self.deltabeta = self.deltabeta.reshape(self.rbf_size);
        #print("deltabeta",self.deltabeta.reshape(4));
        for i in range(self.rbf_size):
            self.deltabeta[i] *=(self.rbf(i)*self.norm2(i)*2*self.beta[i]);
        self.deltabeta *=(-self.learnrate);

        self.omega +=self.deltaomega;
        self.beta +=self.deltabeta;

myRBF = RBF();

for i in range(200):
    for d in dataset:
        myRBF.update(d[:2],d[2:]);
