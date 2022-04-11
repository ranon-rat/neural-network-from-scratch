import numpy as np 
from json import dumps,loads
from time import time
#https://stackabuse.com/python-how-to-flatten-list-of-lists/



tanh={
    "activation":(lambda x:np.tanh(x)),
    "derivate":np.vectorize(lambda y:1-(y**2))
}

sigmoid={
    "activation":np.vectorize(lambda x:1/(1+np.exp(-x))),
    "derivate":np.vectorize(lambda y:y*(1-y))
}
    
relu={
    "activation":np.vectorize(lambda x:max(0,x)),
    "derivate":lambda y:np.where(y>0,1,0)
}
activation_functions={
    "tanh":tanh,
    "relu":relu,
    "sigmoid":sigmoid

}


class NeuralNetwork():

    def __init__(self,length_of_input:int,length_of_output:int,hidden_layers=[2],learning_rate=0.1):
        
        layers_length=[length_of_input]+hidden_layers+[length_of_output]
        bias=[]
        weights=[]       
        
        funcs=[relu]*len(hidden_layers)+[sigmoid]       
        layers=[[]]*len(layers_length)
        
        for l in range(len(layers_length)-1):
            bias.append(np.matrix(np.random.rand(layers_length[l+1],1)))
            weights.append(np.matrix(np.random.rand(layers_length[l+1],layers_length[l])))

        self.activation_funcs_names=["relu"]*len(hidden_layers)+["sigmoid"]
        self.activation_funcs=funcs
        
        self.layers_length=len(layers_length)
        
        self.layers=layers

        self.weights=(weights)
        self.bias=(bias)

        
        self.learning_rate=learning_rate
        
    
    def copy_structure(self,nn:dict)->None:
       
        
        self.activation_funcs_names=nn["activation_funcs"]
        #im just using the names to access the functions
        self.activation_funcs=list(map(lambda x:activation_functions[x],nn["activation_funcs"]))
        # i make the same thing that i make when i define the class
        self.layers_length=nn["layers_amount"]
     

        # i just copy the weights and bias
        self.weights=list(map(np.matrix,nn["weights"]))
        self.bias=list(map(np.matrix,nn["bias"]))
    
    def give_structure(self)->dict:
        return {  
            "max_learning_rate":self.max_lr,
            "min_learning_rate":self.min_lr,
            "activation_funcs":self.activation_funcs_names,
            "layers_amount":self.layers_length,
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

    def predict(self,input:list)->np.matrix:
       
        layers=self.layers
        weights=self.weights
        bias=self.bias
        layers[0]=np.matrix(input).transpose()
        for l in range(len(layers)-1):
            """
            http://matrixmultiplication.xyz/
            """
            res=np.dot(weights[l],layers[l])
            res=np.add(res,bias[l])
            layers[l+1]=self.activation_funcs[l]["activation"](res)
     
       
        self.layers=layers
       
        return layers[-1]



      
    
    def gen_loss(self,train:list)->float:
        # this is for getting the general cost
        # its ugly
        loss=sum(map(lambda x:self.loss(x["target"],x["input"])[0] ,train))
        return loss/len(train)
    def root_gen(self,target,input)->np.matrix:
        c,_=self.loss(target,input)
        x=np.mean(c)
        x=np.sqrt(x)
    def loss(self,target:float,input:float)->float:

        output=self.predict(input)
        #its a lisp reference omg
        return (np.sum(np.square((np.subtract(output,(target))))),output)



    def train(self,target,input)->None:
        # this is for updating the learning rate
        # with this the probability of getting stuck in local minima
        # is reduced
       
        
    
        self.backprop(target,input) 

        
    def clear_layers(self)->None:

        self.layers=[[]]*self.layers_length
    def gradient_descent(self,output:np.matrix,errors:np.matrix,derivate_func)->np.matrix:
        gradient=derivate_func(output)
        gradient=np.multiply(gradient,errors)
        gradient=(np.multiply(gradient,self.learning_rate))
        return gradient
       
    def backprop(self,target:float,input:float):
        
        output=self.predict(input)
        target=np.matrix(target).transpose()
        errors=np.subtract(target,output)        
        gradient=self.gradient_descent(output,errors,self.activation_funcs[-1]["derivate"])

        for l in reversed(range(len(self.layers)-1)):
   
            layers_t=(self.layers[l]).transpose()
            deltgrad=np.dot(gradient,layers_t)
            # this is for getting the weights.. you know what i mean  
             # its the delta of the gradient so 
            # it will reduce the errors

            self.weights[l] =np.add(self.weights[l],deltgrad)
            self.bias[l]    =np.add(self.bias[l],gradient)
            
            if l==0:break    

            weight_t=(self.weights[l]).transpose()
            errors=np.matmul(weight_t,errors)
            gradient=self.gradient_descent(self.layers[l],errors,self.activation_funcs[l-1]["derivate"])
     

