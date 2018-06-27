'''
Author Matti Keskiniemi
Version 16.5.2018
Initialisation for our Neural Network. This script will load MNIST data, train our NN and eventually tell how badly we went wrong :D
Currently it assumes that you have MNIST data right with your code. It can easily be changed.
TODO: Improve code readibility and assign functions instead of straight script
'''
import NeuralNetwork
import numpy
import time

#To measure time of completion
start_time=time.time()
#Variables for our NN
in_nodes=784
hidden_nodes=100
out_nodes=10
learning_rate=0.3


def open_file(file_to_open):
    file=open(file_to_open)
    data=file.readlines()
    file.close()
    return data

#opening training data
training_data=open_file("mnist/mnist_train.csv")
test_data=open_file("mnist/mnist_test.csv")
nn=NeuralNetwork.neural_network(in_nodes,hidden_nodes,out_nodes,learning_rate)



#a training cycle. We will cycle through every item  
cycle_counter=1
for data in  training_data:
    one_data_set=data.split(",")
    inputs=(numpy.asfarray(one_data_set[1:])/ 255.0 * 0.99) + 0.01
    out_targets= numpy.zeros(out_nodes)+0.01
    out_targets[int(one_data_set[0])] = 0.99
    nn.train(inputs,out_targets,cycle_counter)
    cycle_counter+=1


    #test cycle
one_data_set= test_data[0].split(",")
result= nn.query( (numpy.asfarray(one_data_set[1:]) /255.0 * 0.99) + 0.01 )
print(result)
print("aikaa kului " + str( (time.time() - start_time)*3.60 ) +" sekuntia"  )
    



