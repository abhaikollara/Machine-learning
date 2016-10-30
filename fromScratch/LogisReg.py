
# coding: utf-8

# In[1]:

import numpy as np
np.set_printoptions(suppress=True)


# In[15]:

#Import data
train_features=np.loadtxt(fname="../Kaggle/datasets/Iris2.csv",skiprows=1,delimiter=',',usecols=(1,2,3,4),dtype='float64')
train_labels=np.loadtxt(fname="../Kaggle/datasets/Iris2.csv",skiprows=1,delimiter=',',usecols=(5,),dtype='float64')
train_labels = train_labels.reshape(-1, 1)


# In[9]:

def sigmoid(z):
    return (1/(1+np.exp(-z)))


# In[103]:

class LogisticRegression:
    
    def __init__(self):
        self.ALPHA = 0.5

    def scale(self, data):
        denominator = np.amax(data, axis=0)
        # print (data/denominator)
        return (data/denominator), denominator
    
    def train(self, X, y):
        self.N_ITEMS = X.shape[0]
        X = np.concatenate((np.ones(self.N_ITEMS).reshape(-1,1),X),axis=1) #Adding bias term
        self.weights = abs(np.random.randn(X.shape[1]).reshape(1,-1))
        X, self.xdenom = self.scale(X)
        y, self.ydenom = self.scale(y)
        #Gradient descent
        for i in range(1000):
            difference = self._predict(X)-y
            cost = np.sum(y*np.log(self._predict(X))-(1-y)*np.log(1-self._predict(X)))/(self.N_ITEMS)  #Computing cost function
            self.weights = self.weights - (self.ALPHA/self.N_ITEMS)*(np.sum((difference*X), axis=0)) #Performing gradient descent
            #print self.weights #Uncomment to see progression of weights

    def _predict(self, X):
        return sigmoid(np.matmul(X, self.weights.reshape(-1,1)))

    def predict(self, X):
        scaled = X/self.xdenom
        return (self._predict(scaled))


# In[104]:

test = np.array([[1,5,3.3,1.4,0.2]],dtype="float64")

model = LogisticRegression()
model.train(train_features,train_labels)

ans=model.predict(test)

print ans

