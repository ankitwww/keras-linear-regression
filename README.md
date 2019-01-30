# A basic example of using keras
Create a 100 data points and we will fit them on a straight line.


## Building the model
* Steps to build a keras model: 
    * Define the model
    * Add layers
    * Compile the model
    * Train the model
* We create a sequential model, which is the basic model in keras. 
* Only a single connection is required, so we use a Dense layer with linear activation.
* Take input x and apply weight, w, and bias, b (wx + b ) followed by a linear activation to produce output.
* Train the weights for 200 epochs. The value of weights should become 4.
* Define mean squared error(mse) as the loss with simple gradient descent(sgd) as the optimizer.
* get_weights() : returns the weights of the layer as a list of Numpy arrays. 
* summary() :  prints a summary representation of your model. 
