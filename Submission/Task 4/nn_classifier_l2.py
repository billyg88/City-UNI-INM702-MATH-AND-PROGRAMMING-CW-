import pickle
import random # for shuffling the dataset at the start of an epoch 


"""
This is an implementation of a NN class, with RELU/Linear activation functions, and L2 regularization of weigths. 
 The weight update is a bit different from 
the simple NN.

"""


"""
#IMPORT MNIST CODE
with open("Datasets/processed/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size


# Import SHVN DATASET

"""
with open("Datasets/processed/pickled_shvn.pkl", "br") as fh:
    data = pickle.load(fh)
    
train_imgs = data[0]
test_imgs = data[3]
train_labels_one_hot = data[1]
test_labels_one_hot = data[4]
train_labels = data[2]
test_labels = data[5]






import numpy as np
import matplotlib.pyplot as plt
#from scipy.special import softmax


#set seed for reproducible results

np.random.seed(8)
np.random.RandomState(8)



class billy_nn:
    
      
      
      
    def linear_activation_batch(self,matrix):
          
          return matrix

      
    def relu_activation_batch(self,matrix):
          
          
          return np.maximum(0,matrix)
    
      
    def relu_derivative_batch(self, matrix):
          
        matrix[matrix<=0] = 0
        matrix[matrix>0] = 1
        
        return matrix
  
      

      
    def softmax_activation_batch(self, matrix):
          
        z = matrix - np.max(matrix, axis=-1, keepdims=True) #prevent overflow here, with this 
        numerator = np.exp(z)
        denominator = np.sum(numerator,1)
        denominator = denominator.reshape(matrix.shape[0],-1) # (number of samples, 1)
        
        probs = numerator/denominator
        
        return probs      
          



    
    
    def __init__(self, architecture = [1024, 100, 10] , bias = False, activation = 'RELU',  learning_rate = 0.0015, 
                 regularizer_l2 = False, L2_term = 0.005):
        
        self.bias = bias
        
        self.activation = activation
        
        self.architecture = architecture
        
        self.learning_rate = learning_rate
        
        self.regularizer_l2 = regularizer_l2
        
        self.L2_term = L2_term
        
        self.initialize_weights() #initialize weights by taking into account the architecture
        
      
        
    def initialize_weights(self):
            
            self.weights = []
            self.biases = []
            
            #initialize weights for arbitrary lenght NN 
            
            for _ in range(len(self.architecture)-1):
                
                weight_matrix = np.random.normal(loc=0.0,scale=2/np.sqrt(self.architecture[_]+self.architecture[_+1]),
                                                 size=(self.architecture[_],self.architecture[_+1]))
                
                self.weights.append(weight_matrix)
                
                #biases = np.random.normal(loc=0.0, scale=1,size=(self.architecture[i+1]))



  
    def calculate_cost_batch(self, probs, labels):
          
          losses = labels * np.log(probs+ 1e-5) # works against underflow
          
          #losses
          
          batch_loss = - losses.sum()
          
          return batch_loss


        
        
    def train_on_batch(self, batch_samples, batch_labels):
        
        
                
          batch_probs, hidden_activations = self.forward_batch_propagation(batch_samples)
         
          #calculate batch loss      
          batch_loss = self.calculate_cost_batch( batch_probs, batch_labels )
          self.batch_loss = batch_loss       
                
          ####update weights for the batch, first backpropagate the error, and then update each weight matrix
                
          self.update_weights_batch( batch_probs, hidden_activations, batch_labels, batch_samples )
               
                
          return True       
                
                
                
                
                

          
    def forward_batch_propagation(self, batch_samples):
        
        # propagate the batch signal through the network, not using biases 
        input_batch = batch_samples
        hidden_activations = [] # needed for gradient calculation
        
        for weight in self.weights:
            
            trans_batch = np.dot(input_batch, weight) #matrix multiplication, no biasses added
            
            if weight.shape[1] == 10: #if we are multipying the by the final weight matrix
                #apply softmax activation to the batch
                probabilities_batch = self.softmax_activation_batch(trans_batch)
                break
                
                
            elif self.activation == 'RELU':   
                
                output_batch = self.relu_activation_batch(trans_batch)
                hidden_activations.append(output_batch)

            input_batch = output_batch
                
            
        #logits_batch = np.dot(batch_samples, self.weights)
        
        #probabilities_batch = self.softmax_activation_batch(logits_batch)
        
        #print('Probs ',probabilities)
        
        #print (' \n Probs sum', probabilities.sum() )
        
        return probabilities_batch, hidden_activations
                  
        
  
      
      
      
      
     # back-propagation and update of the weights of the Neural Network 
     #
     # back-propagation of loss from the ouput layer using the Transpose of the weights 
     
     #
     # update of the weights using the hidden activation at each layer 
     # 
     #
     #
     
     
    def update_weights_batch(self, batch_probs, hidden_activations, batch_labels, batch_samples) :
          
          hidden_activations.reverse()
          
          output_layer_error = batch_probs - batch_labels # error to propagate 
          
          weights_list = list(self.weights)
          weights_list.reverse()
          
          
          # back-propagation of the error along with the derivative of the activation function (relu)
          # the errors include the the derivatives of the activation functions
          
          layer_errors = []  # reverse this if needed
          
          layer_errors.append(output_layer_error.T)
         
          
          error_l = output_layer_error
          
          # for an NN with 2 hidden layers, 2 back-propagation methods will be run
          
          # add clause for linear and sigmoid derivative apropriatly.
          
          # back-prop of the eror to every layer in the network 
          for i in range(len(weights_list)-1):
                
                error_term = np.dot(weights_list[i],error_l.T)
                
                if self.activation == 'RELU':
                
                      derivative_term = self.relu_derivative_batch(hidden_activations[i].T)
                                            
                
                #element-wise multiplication for the full error expression
                error_l_minus = error_term * derivative_term
                
                layer_errors.append(error_l_minus)
                
                error_l = error_l_minus.T
                
                
          # layer errors created here. 
          
          # update weights here using the layer errors and the hidden activations 
          activations = list(hidden_activations)
          activations.reverse()
          
          activations.insert(0,batch_samples)
          activations.reverse()

          #activations.append(batch_probs)
          
          # check for possible regularization here 
          #weights_list.reverse()
          
          for i in range(len(layer_errors)):
                
                #self.weights[i] or self.weights = reverse_weight_list
                
                weight_update = np.dot(layer_errors[i],activations[i])
                
                #creating the regularized update  
                # 
                weight_update = weight_update +  (self.L2_term * weights_list[i].T)
                
                
                weights_list[i] -= self.learning_rate * weight_update.T #take some of the gradient using learning rate
                
          
          
          #final update, update network parameters to new weights
          #
          #
          weights_list.reverse()
          
          self.weights = weights_list

      
      
      
    def evaluate(self,data,labels):
        
        corrects, wrongs = 0, 0
        
        for i in range(len(data)):
              
            res = self.infer_sample(data[i])
            #sumaaaaaaaaaaaa = res.sum()
            res_max = res.argmax()
            #ruin=labels[i]
            
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs    




    def infer_sample(self,sample):
        #use this function to get a sample prediction, after the network has been trained   
        prediction = self.signal_propagation_test(sample)
          
        return prediction  
  
   
    def signal_propagation_test(self,sample):
          
          # get a prediction-class, for a sample vector, forward propagation
                    
          trans_vector = sample
          
          for weight in self.weights:
                
                trans_vector = np.dot(trans_vector, weight) # matrix transformation 
                
                trans_vector = self.relu_activation_batch(trans_vector) # relu activation, no bias added
                
          prediction = trans_vector 
          
          return prediction
                
            
            
            
            
            
### Modelling

nn = billy_nn( architecture = [1024,200,200,100,10], bias = False, activation = 'RELU',  
              learning_rate = 0.0001, regularizer_l2 = True,  L2_term = 0.7)     # standard call gets you an MLP. So only 1 hidden layer.

print('\n')
weights = nn.weights

#print('NN weights ', nn.weights.shape," weights: "+str(nn.weights.shape[0]+nn.weights.shape[1])+ ' \n')




### Test the network with no training 
corrects, wrongs = nn.evaluate(test_imgs, test_labels)
print("\n  Testing accuracy before TRAINING ", corrects / ( corrects + wrongs), '\n')





# Mini-batch gradient descent 
batch_size = 256 #size of the mini-batch
epochs = 20 
iteration_losses = [] #should be the same len aas the number of iteratios
epoch_accuracies = []
epoch_costs = [] # should match the number of epochs set


#Print training elements
num_of_batches = round(train_imgs.shape[0] / batch_size) + 1
num_of_iterations = epochs * num_of_batches

print ('\n Batch size',batch_size )
print('_________________________________________________')


print ('\n Number of batches per epoch',num_of_batches )
print('_________________________________________________')

print ('\n Number of iterations to perform:',num_of_iterations )
print('_________________________________________________')
print('_________________________________________________\n')


print ("Started regularized training")



#for each epoch 
for epoch in range(epochs):

            # cycle through all minibatches of data
            n_samples = train_imgs.shape[0]
            #shuffle entire dataset indices for proper mini-batch GD
            indices = np.arange(train_imgs.shape[0])
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, batch_size):
            
                  end = min (start + batch_size, n_samples)
                  batch_indices = indices[start:end]
                  
                  #train nn on mini-batch data
                  nn.train_on_batch(train_imgs[batch_indices],train_labels_one_hot[batch_indices])
                  
                  #get mnin-batch cross-entropy loss term 
                  cel = nn.batch_loss
                  
                  #create l2 term for loss calculation
                  #l2_cost = (nn.L2_term / 2) * np.sum(np.square(nn.weights))
                  
                  weight_squared_sums = [np.sum(np.square(weight_matrix)) for weight_matrix in nn.weights]
                  
                  l2_cost = (nn.L2_term / 2) * np.array(weight_squared_sums).sum()
            
                  overall_cost = cel + l2_cost                       
                  
                  #save loss on the mini-batch
                  iteration_losses.append(overall_cost)
                  
            epoch_loss = iteration_losses[-num_of_batches:] # GET THE LAST X ITERTIONS THAT MAKE UP THE EPOCH 
            epoch_costs.append(np.array(epoch_loss).sum())

            
            #Evaluate training accuracy after each iteration
            corrects, wrongs = nn.evaluate(train_imgs, train_labels) #this is the integer representation
            accu = corrects / ( corrects + wrongs)
            print('_________________________________________________\n')
            print("Training accuracy after epoch ", accu, '\n')
            epoch_accuracies.append(accu)
            
            #epoch completed 
            print ("Epochs completed {} / {} ".format(epoch+1,epochs))

      



plt.plot(range(num_of_iterations), iteration_losses)
plt.ylabel('Cost')
plt.xlabel('Iterations')
#plt.savefig('loss.png', dpi=300)
plt.show()




plt.plot(range(epochs), epoch_accuracies)
plt.ylabel('Accuracy %')
plt.xlabel('Epochs')
#plt.savefig('loss.png', dpi=300)
plt.show()


plt.plot(range(epochs), epoch_costs)
plt.ylabel('Cost')
plt.xlabel('Epochs')
#plt.savefig('loss.png', dpi=300)
plt.show()




### Test the network with no training 
corrects, wrongs = nn.evaluate(test_imgs, test_labels)
print("\n  Testing accuracy after training ", corrects / ( corrects + wrongs), '\n')












