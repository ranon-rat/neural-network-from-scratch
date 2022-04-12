import numpy as np
from random import choice, shuffle
from src.neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import pandas as pd
nn=NeuralNetwork(784,10,hidden_layers=[16,16],learning_rate=0.07)
_,ax=plt.subplots(2)
def main():
    data=pd.read_csv("data/train.csv")
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and 
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.
    accuracy=nn.train(X_dev,Y_dev,500,len(data_dev))
    nn.save(name="nn.json")
    ax[0].plot(accuracy)
    ax[0].set_title("accuracy")
    ax[0].set_xlabel("epochs")
    ax[0].set_ylabel("accuracy")
    x= X_dev[:, 32, None]
    y=Y_dev[32]
    out=nn.prediction(x)
    ax[1].imshow(x.reshape(28,28),cmap="gray")
    ax[1].set_title("the actual number"+str(y) +" vs the predicted number "+str(out))
   

    


    plt.show()
if __name__ == "__main__":
    main()