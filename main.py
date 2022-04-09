from random import choice
from src.neural_network import NeuralNetwork

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
def test(nen):
    
    print("it should be near to 1     ",nen.predict([1,0]),nen.cost([1,0],[1]))
      
    print("the same as before,1       ",nen.predict([0,1]),nen.cost([0,1],[1]))
  
    print("it should be near to 0     ",nen.predict([1,1]),nen.cost([1,1],[0]))
    
    print("the same as before, nothing",nen.predict([0,0]),nen.cost([0,0],[0]))

   
def main():

   
   
    nen=NeuralNetwork(2,1,[2])

  
    test(nen)
 
 
    for i in range(6000):
        t=choice(train)
        nen.train(t['input'],t['target'])
        nen.clear_nn()  
        
    test(nen)

 
if __name__ == "__main__":
    main()
