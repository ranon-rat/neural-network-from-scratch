
from random import choice
from src.neural_network import NeuralNetwork
from matplotlib import pyplot as plt

from celluloid import Camera


plt.style.use('dark_background')

fig=plt.figure()
camera=Camera(fig)
nen=NeuralNetwork(2,1,[2],min_lr=1e-3,max_lr=1e-1,decay_factor=0.99)

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
    print(("#"*10),"TESTING #",testing,("#"*10))
    for t in train:
        print("\nTESTING",t['input'],"TARGET",t['target'])
        print("\ninput:",t['input'],"target:",t['target'])
        cost,output=nen.cost(t['input'],t['target'])
        print("prediction:",output)
        print("cost:",cost,"\n")
        print("-"*20)
    
   
    testing+=1
    print("GENERAL COST:",nen.general_cost(train))
    
def main():
    test(nen)
    epochs=1000
    y=[0]*(epochs)
    

    for i in range(epochs):
        t=choice(train)
        nen.train(epochs,i,t['input'],t['target'])
       
        cost=nen.general_cost(train)
        #=nen.general_cost(train)
        y[i]=(cost)
    plt.plot(y)
    test(nen)

       
    plt.show()
        
       
     #   camera.snap()
    #ani=camera.animate()
    #ani.save('animation.gif', writer='ffmpeg', fps=35)
    

    

    
   


 
if __name__ == "__main__":
    main()
