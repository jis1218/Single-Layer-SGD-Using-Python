# coding: utf-8

from dataset.mnist import load_mnist
import numpy as np

def softmax(y):
    y = y.T
    y = y - np.max(y, axis=0)
    output = np.exp(y)/np.sum(np.exp(y), axis=0)
    
    return output.T

def lossFunc(y, t):
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))/y.shape[0]

def accuracy(y, t):
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    
    accuracy = np.sum(y==t)/float(y.shape[0])
    return accuracy

if __name__ == '__main__':
    
    learning_rate = 0.1
    
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    
    train_size = x_train.shape[0]
    batch_size = 100
   
    
    W = 0.01*np.random.randn(x_train.shape[1], np.shape(t_train)[1])

    momentum_rate = 0.9
    momentum = 0
    
    for i in range(10000):
        batch_mask = np.random.choice(train_size, batch_size)

        x_batch = x_train[batch_mask]

        t_batch = t_train[batch_mask]
        
        y = np.dot(x_batch, W)
        output1 = softmax(y)
        dx = (output1-t_batch)/batch_size
        loss = lossFunc(output1, t_batch)
        
                
        if i%1==0:
            #print(loss)
            y_test = np.dot(x_test, W)
            output2 = softmax(y_test)
            #print(accuracy(output1, t_batch), accuracy(output2, t_test))
            if accuracy(output2, t_test)>0.90:
                print(i)
                break;
                
            
            
#         momentum = momentum*momentum_rate - learning_rate*np.dot(x_batch.T, dx) 
#         W = W + momentum
        W = W - learning_rate*np.dot(x_batch.T, dx)
    
    pass



