#!/usr/bin/env python
# coding: utf-8

# ## Import packages
# * numpy - package for scientific computing with Python
# * pyplot - provides a MATLAB-like plotting framework.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# ## Import keras packages
# * keras - the neural network library
# * Sequential -  basic keras model composed of a linear stack of layers.
# * Dense - a regular densely-connected NN layer.

# In[3]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# ## Define a linear regression problem
# * Create a 100 data points and we will fit them on a straight line.
# * train_data_X has values between –1 and 1, and train_data_Y has 4 times the train_data_X and some randomness.

# In[4]:


train_data_X = np.linspace(-1, 1, 101)
train_data_Y = 4 * train_data_X + np.random.randn(101) * 0.44


# In[5]:


train_data_X


# In[6]:


train_data_Y


# In[7]:


plt.scatter(train_data_X, train_data_Y, label='data',color=['red','green'])


# ## Building the model
# * Steps to build a keras model: 
#     * Define the model
#     * Add layers
#     * Compile the model
#     * Train the model
# * We create a sequential model, which is the basic model in keras. 
# * Only a single connection is required, so we use a Dense layer with linear activation.
# * Take input x and apply weight, w, and bias, b (wx + b ) followed by a linear activation to produce output.
# * Train the weights for 200 epochs. The value of weights should become 4.
# * Define mean squared error(mse) as the loss with simple gradient descent(sgd) as the optimizer.
# * get_weights() : returns the weights of the layer as a list of Numpy arrays. 
# * summary() :  prints a summary representation of your model. 

# In[8]:


model = Sequential()


# In[9]:


model.add(Dense(input_dim=1,output_dim=1, init='uniform', activation='linear'))


# In[10]:


# model.add(Dense(input_dim=1, name='Layer_1', output_dim=1, init='uniform', activation='linear'))


# In[11]:


model.summary()


# In[12]:


len(model.layers)


# In[13]:


weights = model.layers[0].get_weights()


# In[14]:


len(weights)


# In[15]:


weights


# In[16]:


weight_initial = weights[0]


# In[17]:


bias_initial = weights[1]


# In[18]:


print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (weight_initial, bias_initial))


# In[19]:


model.compile(optimizer='sgd', loss='mse')


# In[20]:


model.fit(train_data_X, train_data_Y, epochs=200, verbose=1)


# In[21]:


weights = model.layers[0].get_weights()


# In[22]:


weight_final = weights[0]


# In[23]:


bias_final = weights[1]


# In[24]:


print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (weight_final, bias_final))


# ## Test the model
# * predict() - Generates output predictions for the input samples.
# * result will be weight_final * train_data_X + bias_final
# * plot the regression line.

# In[25]:


result = model.predict(train_data_X)


# In[26]:


plt.scatter(train_data_X, train_data_Y, label='data',color=['red','green'])
plt.plot(train_data_X, result, label='prediction')
plt.legend()
plt.show()


# In[ ]:




