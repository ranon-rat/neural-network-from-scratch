import numpy as np 
from json import dumps,loads
from time import time
#https://stackabuse.com/python-how-to-flatten-list-of-lists/

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

tanh={
    "activation":np.vectorize(lambda x:np.tanh(x)),
    "derivate":np.vectorize(lambda y:1-(y**2))
}

sigmoid={
    "activation":np.vectorize(lambda x:1/(1+np.exp(-x))),
    "derivate":np.vectorize(lambda y:y*(1-y))
}
    
relu={
    "activation":np.vectorize(lambda x:max(0,x)),
    "derivate":np.vectorize(lambda y:np.where(y>0,1,0))
}

soft_max={
    "activation":np.vectorize(lambda x:np.exp(x)/np.sum(np.exp(x))),
    "derivate":np.vectorize(lambda y:0 - (y[0] * y[1]))
}
activation_functions={
    "tanh":tanh,
    "relu":relu,
    "sigmoid":sigmoid,
    "soft_max":soft_max,
}






class NeuralNetwork():

    def __init__(self,length_of_input:int,length_of_output:int,hidden_layers=[2],learning_rate=0.1):
        
        layers_length=[length_of_input]+hidden_layers+[length_of_output]
        bias=[]
        weights=[]       
        
        funcs=[relu]*len(hidden_layers)+[sigmoid]       
       
        
        for l in range(len(layers_length)-1):
            bias.append((np.random.rand(layers_length[l+1],1))-0.5)
            weights.append((np.random.rand(layers_length[l+1],layers_length[l]))-0.5)

        self.activation_funcs_names=["relu"]*len(hidden_layers)+["sigmoid"]
        
        self.activation_funcs=funcs
        
        self.layers_amount=len(layers_length)
       
        self.weights=(weights)
        self.bias=(bias)

        
        self.learning_rate=learning_rate
        
    
    def copy_structure(self,nn:dict)->None:
       
        
        self.activation_funcs_names=nn["activation_funcs"].copy()
        #im just using the names to access the functions
        self.activation_funcs=list(map(lambda x:activation_functions[x],nn["activation_funcs"])).copy()
        # i make the same thing that i make when i define the class
        self.layers_amount=nn["layers_amount"]
     

        # i just copy the weights and bias
        self.weights=list(map(np.array,nn["weights"])).copy()
        self.bias=list(map(np.array,nn["bias"])).copy()
    
    def give_structure(self)->dict:
        return {  
          
            "activation_funcs":self.activation_funcs_names,
            "layers_amount":self.layers_amount,
            "weights":list(map(lambda x:x.tolist(), self.weights)),
            "bias":list(map(lambda x:x.tolist(), self.bias)),
           
        }
    def copy_from(self,path:str):
        with open(path, "r") as infile:
            nn=loads(infile.read())
            self.copy_structure((nn))
        
    def save(self,path:str="",name="neural_network"+str(time())+".json")->None:
        with open(path+name, "w") as outfile:
            s=dumps(self.give_structure(),indent = 4)
            outfile.write(s)

    def feed_foward(self,input)->np.matrix:
       
        
        layers=[0]*self.layers_amount
        
        layers[0]=input

        for l in range(self.layers_amount-1):
            """
            http://matrixmultiplication.xyz/
            """
            res=self.weights[l].dot(layers[l])+self.bias[l]
            layers[l+1]=self.activation_funcs[l]["activation"](res)
     
          
        return layers



      
    
  
    def loss(self,target,prediction)->float:

     
        
        return np.sum((prediction-target)**2)
    #just for testing
    def get_predictions(self,A2):
        return np.argmax(A2, 0)
    def prediction(self,x):
        l=self.feed_foward(x)
        out=l[-1]
        return self.get_predictions(out)
    # more info
    def get_accuracy(self,predictions, Y):
        return np.sum(predictions == Y) / Y.size
    
    def update(self,dw,db):
        for i in range(len(self.weights)):
            self.weights[i]=(self.weights[i ]-(dw[i]*self.learning_rate))
            self.bias[i]=   (self.bias[i]-( db[i]*self.learning_rate))

    def train(self,X,Y,iterations,m):
        accuracy_log=[]
        loss_log=[]
        for i in range(iterations):
            layers=self.feed_foward(X)

            dw,db=self.backprop(layers,Y,m) 
            self.update(dw,db)

            
            if i%10==0:
                print("_"*30,i,"_"*30)
                
                predictions=self.get_predictions(layers[-1])
                accuracy=self.get_accuracy(predictions,Y)
                loss=self.loss(Y,predictions)
                accuracy_log.append(accuracy)
                loss_log.append(loss)
                
                
                print("| predictions:",predictions[:10],"| accuracy:",str(accuracy)[:10],"|","loss:",str(loss)[:10],"|")
             
                
        return accuracy_log,loss_log

    
    

    def backprop(self,layers,target,m):
        
        
        one_hot_Y = one_hot(target)
        errors=layers[-1]-one_hot_Y
        gradient=self.activation_funcs[-1]["derivate"](layers[-1])*errors


        dw=[0]*(len(self.weights))
        db=[0]*(len(self.bias))
        for l in range(len(layers)-1)[::-1]:
   
            layers_t=layers[l].transpose()
            deltgrad=np.dot(gradient,layers_t)
            # this is for getting the weights.. you know what i mean  
             # its the delta of the gradient so 
            # it will reduce the errors

            dw[l] =(1 / m) *deltgrad
            db[l] =(1 / m) *np.sum(gradient)

            if l==0:break    
            errors=self.weights[l].T.dot(errors)
            gradient=(errors*self.activation_funcs[l-1]["derivate"](layers[l]))

        return (dw,db)