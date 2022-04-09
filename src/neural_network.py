
import numpy as np 
#https://stackabuse.com/python-how-to-flatten-list-of-lists/
flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]
sigmoid= lambda x: 1/(1+np.exp(-x))
dsigmoid=np.vectorize(lambda y:y * (1 - y))


class NeuralNetwork():

    def __init__(self,length_of_input,length_of_output,hidden_layers=[2]):
        layers=[length_of_input]+hidden_layers+[length_of_output]
        bias=[]
        weights=[]
        nn=[[]]*len(layers)
        for l in range(len(layers)-1):
            bias.append(np.matrix(np.random.rand(layers[l+1],1)))
            weights.append(np.matrix(np.random.rand(layers[l+1],layers[l])))
       
        self.nn=(nn)
        self.weights=(weights)
        self.bias=(bias)
        self.learning_rate=4

    def feed_foward(self,input):
        nn=self.nn 
        weights=self.weights
        bias=self.bias
        """
        for some reason , after i convert the input to a matrix
        it returns me [[x],[x]] instead of
        """
        nn[0]=np.matrix(input).transpose()
      
        for l in range(len(nn)-1):
            
            """
         
            idk how to explain myself so , here is a good resource 
            http://matrixmultiplication.xyz/
            """
            nn_t=nn[l]
            res=np.matmul(weights[l],nn_t)
            nn[l+1]=sigmoid(res+bias[l])
      
        output=nn[-1]
        self.nn=nn
        
        
        return output



      
    def clear_nn(self)->None: #yes , i hate this too
        for l in range(len(self.nn)):
            self.nn[l]-=self.nn[l]
  
    def cost(self,input,target):
        output=self.predict(input)
        return (np.sum(np.square(output-target)),output)

    def predict(self,input):
        output=self.feed_foward(input)
        return output   
    def gradient_descent(self,output,errors):
        gradient=(dsigmoid(output))
        # then i need to multiply it for gettint the erros and with that
        # i will know how to reduce them

        gradient=np.multiply(gradient,errors)
        # the learning rate its just for the speed of the training
        # maybe it could be more inexact and have more errors so
        # if you wanted you can change it
        # but maybe it will be in a point where it would be less presice
        # local minima
        gradient=np.multiply(gradient,self.learning_rate)
        return (gradient)
    def train(self,input,target):
        
        output=self.feed_foward(input)
       
        errors=(np.subtract(target,output))
        gradient=self.gradient_descent(output,errors)
        l=len(self.nn)-2
        while l>=0:
            #print("gradient",gradient,self.nn[l])
            nn_t=np.transpose(self.nn[l])
          
            # this is for getting the weights.. you know what i mean
            deltGrad=( np.matmul(gradient,nn_t))
            # its the delta of the gradient so 
            # it will reduce the errors
         
            self.weights[l]=np.add(self.weights[l], deltGrad)
           
            self.bias[l]=np.add(self.bias[l],gradient)
            if l==0:break
        
            weight_t=np.transpose(self.weights[l])
            errors=np.matmul(weight_t,gradient)
            gradient=self.gradient_descent(self.nn[l],errors)
            l-=1
        self.clear_nn()

      
