import numpy as np
np.set_printoptions(suppress=True)

#Import data
train_features=np.loadtxt(fname="../datasets/linearReg.csv",delimiter=',',usecols=(0,1,2,3),dtype='float64')
train_labels=np.loadtxt(fname="../datasets/linearReg.csv",delimiter=',',usecols=(4,),dtype='float32')
train_labels = train_labels.reshape(-1, 1)


class LinearRegression:
    
    def __init__(self):
        self.ALPHA = 0.5

    def scale(self, data):
        denominator = np.amax(data, axis=0)
        # print (data/denominator)
        return (data/denominator), denominator
    
    def train(self, X, y):
        self.N_ITEMS = X.shape[0]
        X = np.concatenate((np.ones(self.N_ITEMS).reshape(-1,1),X),axis=1)
        self.weights = abs(np.random.randn(X.shape[1]).reshape(1,-1))
        X, self.xdenom = self.scale(X)
        y, self.ydenom = self.scale(y)
        #Gradient descent
        for i in range(100):
            difference = self._predict(X)-y
            cost = np.sum(np.square(difference))/(2*self.N_ITEMS)   #Computing cost function
            self.weights[:,1:] = self.weights[:,1:] - (self.ALPHA/self.N_ITEMS)*(np.sum((difference*X[:,1:]), axis=0)) #Performing gradient descent
            self.weights[:,0] = self.weights[:,0] - (self.ALPHA/self.N_ITEMS)*(np.sum((difference), axis=0)) #Performing gradient descent
            print self.weights #Uncomment to see weights improving

    def _predict(self, X):
        return np.matmul(X, self.weights.reshape(-1,1))

    def predict(self, X):
        scaled = X/self.xdenom
        return self._predict(scaled)*self.ydenom

test = np.array([[1,2,2,2,2]],dtype="float32")

model = LinearRegression()
model.train(train_features,train_labels)

ans=model.predict(test)

print ans