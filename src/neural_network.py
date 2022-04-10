from math import atan
import numpy as np 
#https://stackabuse.com/python-how-to-flatten-list-of-lists/

sigmoid= np.vectorize(lambda x: 1/(1+np.exp(-x)))
dsigmoid=np.vectorize(lambda y:y * (1 - y))
relu=np.vectorize(lambda x: max(0,x))
drelu=np.vectorize(lambda x:np.where(x>0,1,0))
tanh=np.vectorize(lambda x:np.tanh(x))
dtanh=np.vectorize(lambda y:1-(y**2))
class NeuralNetwork():

    def __init__(self,length_of_input,length_of_output,hidden_layers=[2], min_lr=1e-5,max_lr=1e-2,decay_factor=0.95,step_size=10,cycle_size=10):
        
        layers=[length_of_input]+hidden_layers+[length_of_output]
        funcs=[tanh]*len(hidden_layers)+[tanh]
        dfuncs=[dtanh]*len(hidden_layers)+[dtanh]
       
        bias=[]
        weights=[]
       
        nn=[[]]*len(layers)
    
        for l in range(len(layers)-1):
            bias.append(np.matrix(np.random.rand(layers[l+1],1)))
            weights.append(np.matrix(np.random.rand(layers[l+1],layers[l])))
      
        self.derivate_funcs=dfuncs
        self.activation_funcs=funcs
       
        self.nn=(nn)
        self.weights=(weights)
        self.bias=(bias)
        
        self.min_lr=min_lr
        self.max_lr=max_lr
        self.learning_rate=max_lr
        self.decay_factor=decay_factor

        self.step_size=step_size
        self.batch_size=0
        self.cycle_size=cycle_size

    def feed_foward(self,input):
        nn=self.nn 
        weights=self.weights
        bias=self.bias
        nn[0]=np.matrix(input).transpose()
        for l in range(len(nn)-1):
            
            """
         
            idk how to explain myself so , here is a good resource 
            http://matrixmultiplication.xyz/
            """
        
            res=np.matmul(weights[l],nn[l])
            nn[l+1]=self.activation_funcs[l](res+bias[l])
        
        self.nn=nn
        return nn[-1]



      
    
    def general_cost(self,train):
       
      
        cost=sum(map(lambda x:self.cost(x["input"],x["target"])[0] ,train))
        return cost
    def cost(self,input,target):
        output=self.predict(input)
     
        return (np.sum(np.square(output-target))/len(target),output)

    def predict(self,input):
        output=self.feed_foward(input)
   
        return output   
    #https://www.jeremyjordan.me/nn-learning-rate/
    def update_learning_rate(self):
        x = self.batch_size/(self.step_size*self.cycle_size)
        self.learning_rate=self.min_lr+0.5*(self.max_lr-self.min_lr)*(1+np.cos(x*np.pi))
   
    def gradient_descent(self,output,errors,derivate_func):
        gradient=derivate_func(output)
        gradient=np.multiply(gradient,errors)
        gradient=np.multiply(gradient,self.learning_rate)
        return (gradient)
    

    def train(self,iterations:int,iteration,target,input):
        self.update_learning_rate()
        self.backprop(target,input)
        self.batch_size+=1
       
    def backprop(self,input,target):
        
        output=self.feed_foward(input)
        errors=np.subtract(target,output)
        gradient=self.gradient_descent(output,errors,self.derivate_funcs[-1])
       
        for l in range(len(self.nn)-1)[::-1]:
          
            nn_t=np.transpose(self.nn[l])
            deltgrad=np.matmul(gradient,nn_t)
            # this is for getting the weights.. you know what i mean  
             # its the delta of the gradient so 
            # it will reduce the errors
            
            self.weights[l]=np.add(self.weights[l],deltgrad)
            self.bias[l]=np.add(self.bias[l],gradient)
            if l==0:break    
            weight_t=(self.weights[l]).transpose()
            errors=np.matmul(weight_t,errors)
            gradient=self.gradient_descent(self.nn[l],errors,self.derivate_funcs[l-1])
        


      
