
from random import choice
from src.neural_network import NeuralNetwork
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from celluloid import Camera


plt.style.use('dark_background')

fig=plt.figure()
camera=Camera(fig)
nen=NeuralNetwork(2,1,[2])

train=[
    {
        'input':[0,0],
        'target':[0]
    },
     {
        "input":[1,1],
        "target":[0]
    },
    {
        "input":[1,0],
        "target":[1]
    },
      {
        "input":[0,1],
        "target":[1]
    }, 
]

testing=1


def test(nen):
    global testing,train
    print(("-"*10),"TESTING #",testing,("-"*10))
    for t in train:
        print("\n","TESTING",t['input'],"TARGET",t['target'])
        print("\tinput:",t['input'],"target:",t['target'])
        cost,output=nen.cost(t['input'],t['target'])
        print("\tprediction:",output)
        print("\tcost:",cost)
    
   
    testing+=1
    print("GENERAL COST:",nen.general_cost(train))
    print(nen.nn)
def main():
    test(nen)
    epochs=6000
    y=[0]*(epochs//10)
    

    for i in range(epochs):
        t=choice(train)
        nen.train(t['input'],t['target'])
        
        if i%10==0: 
            cost,_=nen.cost(t['input'],t['target'])
            y[i//10]=(cost)
    plt.plot(y)
    test(nen)

       
    plt.show()
        
       
     #   camera.snap()
    #ani=camera.animate()
    #ani.save('animation.gif', writer='ffmpeg', fps=35)
    

    

    
   


 
if __name__ == "__main__":
    main()
