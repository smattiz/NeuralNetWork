'''
Author Matti Keskiniemi
This is a Common-purpose neural network with three layers.
You may freely fork this to your own projects and improve it as much as you want to.
Dependecies:
-Numpy
-SciPy
-A computer with enough calculation power ;)

'''
import numpy
import scipy.special
class neural_network:
    
    
    def __init__(self, input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.inodes=input_nodes
        self.hnodes=hidden_nodes
        self.onodes=output_nodes
        self.lrate=learning_rate
        #first links weights. Randomised, so we won't get too large or too small starting weights. Also used gaussian distribution to scale weights to +-1
        self.hidden_input_weights= numpy.random.normal(0.0, pow(self.inodes,-0.5), (self.hnodes,self.inodes) ) 
        self.output_hidden_weights= numpy.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes) ) 

        #Activation function using Scipy-library. Consider this as a function pointer, if you are from C# -world
        self.activation_function= lambda x: scipy.special.expit(x)
        
    
    #Actual training
    def train(self, list_of_inputs,target_list, cycle):
        print("Training cycle no. " + str(cycle) )
        #Same as before, convert inputs and targets to numPy-array
        inputs=numpy.array(list_of_inputs,ndmin=2).T
        targets=numpy.array(target_list,ndmin=2).T

        #Calculations done using training examples, also Sigmoid applied
        hidden_inputs=numpy.dot(self.hidden_input_weights,inputs) 
        hidden_outputs=self.activation_function(hidden_inputs)

        final_inputs= numpy.dot(self.output_hidden_weights,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        #Calculate error. Python can apply simple mathematical work on matrices, so we'll just have to subtract one from the other
        output_error=targets-final_outputs
        hidden_error= numpy.dot(self.output_hidden_weights.T,output_error)

        #update weights for the links between hidden and output. We don't apply sigmoid because we have done it earlier
        self.output_hidden_weights += self.lrate * numpy.dot(output_error * final_outputs * (1.0-final_outputs), numpy.transpose(hidden_outputs) )
        self.hidden_input_weights += self.lrate * numpy.dot(hidden_error * hidden_outputs * (1.0- hidden_outputs), numpy.transpose(inputs))

       



    #This is where the fun starts
    def query(self,list_of_inputs):
        #calculate signals which will be fed to hidden layer
        inputs=numpy.array(list_of_inputs,ndmin=2).T
        hidden_inputs=numpy.dot(self.hidden_input_weights,inputs) 
        #activation function applied to link weights given to hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)

        final_inputs= numpy.dot(self.output_hidden_weights,hidden_outputs)
        final_outputs=self.activation_function(final_inputs)

        #To be able to train our NN, we need to return final outputs and compare
        #them to actual results
        return final_outputs