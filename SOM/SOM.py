
#-*- coding:utf-8 -*-

# self organizing map

import numpy as np;



# import matplotlib.pyplot as plt
#
# x = [0, 1]
# y = [0, 1]
# plt.figure()
# plt.plot(x, y)
# plt.savefig("easyplot.jpg");


dataSet = [
        # 1
        [ 0.697, 0.460],
        # 2
        [0.774, 0.376],
        # 3
        [ 0.634, 0.264],
        # 4
        [ 0.608, 0.318],
        # 5
        [ 0.556, 0.215],
        # 6
        [ 0.403, 0.237],
        # 7
        [ 0.481, 0.149],
        # 8
        [ 0.437, 0.211],

        # ----------------------------------------------------
        # 9
        [ 0.666, 0.091],
        # 10
        [ 0.243, 0.267],
        # 11
        [ 0.245, 0.057],
        # 12
        [ 0.343, 0.099],
        # 13
        [ 0.639, 0.161],
        # 14
        [ 0.657, 0.198],
        # 15
        [ 0.360, 0.370],
        # 16
        [ 0.593, 0.042],
        # 17
        [ 0.719, 0.103]
    ]

class SOM:
    def __init__(self):
        self.competitor_size = 4;
        self.input_size = 2;
        self.competitor = np.random.random((self.competitor_size,self.input_size));
        self.competitor = [identify(d) for d in self.competitor];
        self.initsigama = 1;
        self.tao1 = 1;#about topological dis
        self.tao2 = 1;#about learnrate
        self.initlearnrate = 0.9;
    def getwinpoint(self,x):
        minn = 10000000000;
        index = -1;

        for i in range(self.competitor_size):
            t1 = self.competitor[i]-x;
            t2 = np.sum(np.square(t1));
            #print (t2);
            if t2<minn:
                index = i;
                minn = t2;
        assert index != -1;
        return index;
    def getsigamasquare(self,t):
        sigama = self.initsigama*np.exp(-t/self.tao1);
        return sigama*sigama;
    def get_topological_dis(self,j,wp,t):
        return np.exp(-(j-wp)*(j-wp)/(2*self.getsigamasquare(t)))
    def getlearnrate(self,t):
        return self.initlearnrate*np.exp(-t/self.tao2)

    def update(self,inpt,t):
        #wp is winnner point
        wp = self.getwinpoint(inpt);
        #print(wp);
        for i in range(self.competitor_size):
            delta = self.getlearnrate(t)*self.get_topological_dis(i,wp,t)*(inpt-self.competitor[i]);
            self.competitor[i] +=delta;

    def valuation(self,ds):
        for x in ds:
            wp = self.getwinpoint(x);
            print ("data: ",x," win point ",wp);

def identify(x):
    tmp = np.sum(np.square(x));

    return x/np.sqrt(tmp);




mySOM = SOM();
newdata = [ identify(x) for x in dataSet ];
newdata = np.array(newdata);
#print(np.array(newdata));
for i in range(200):
    for d in newdata:
     mySOM.update(d,i);

mySOM.valuation(newdata);

import numpy as np
import matplotlib.pyplot as plt
plt.figure()
def drawline(ds,col='o'):
    x = [d[0] for d in ds];
    y = [d[1] for d in ds];
    print(x);

    plt.plot(x, y);


drawline(mySOM.competitor);
#drawline(newdata[:8],'or');
#drawline(newdata[8:],'ob');

plt.show();