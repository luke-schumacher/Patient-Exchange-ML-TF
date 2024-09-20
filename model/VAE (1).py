#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[48]:


# Load the already encoded data
file_path = "../data/encoded_176398_HEAD.csv"
df = pd.read_csv(file_path)
print(f"Data loaded successfully from {file_path}.\n")
print("Columns in DataFrame:", df.columns.tolist())


# In[49]:


# Drop irrelevant columns
df = df.drop(columns=['datetime','PatientID_from', 'PatientID_to', 'SN', 'BodyPart_from', 'BodyPart_to'])
print("Dropped 'datetime', 'PatientID' and 'BodyPart' columns")


# In[50]:


# Print the shape of the dataframe after dropping columns
print("DataFrame shape after dropping columns:", df.shape)


# In[51]:


# Separate features and target columns
X = df.drop(columns=['BodyGroup_from', 'BodyGroup_to', 'sourceID'])
y_bodygroup_from = df[['BodyGroup_from', 'BodyGroup_to']]
y_sourceid = df['sourceID']
print("Separated features and targets")


# In[52]:


# Limit sourceID to the first 30 unique values
unique_source_ids = y_sourceid.unique()[:30]
y_sourceid = y_sourceid[y_sourceid.isin(unique_source_ids)].reset_index(drop=True)


# In[54]:


# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
# Print the scaled feature shape
print("Scaled features shape:", X_scaled.shape)


# In[55]:


# Define the VAE architecture
input_dim = X_scaled.shape[1]
latent_dim = 2  # Latent space dimension


# In[56]:


# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(64, activation='relu')(inputs)
h = Dense(32, activation='relu')(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


# In[57]:


# Reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])


# In[58]:


# Decoder
decoder_h = Dense(32, activation='relu')
decoder_mean = Dense(64, activation='relu')
outputs = Dense(input_dim, activation='sigmoid')(decoder_mean(decoder_h(z)))


# In[59]:


# VAE model
vae = Model(inputs, outputs)


# In[60]:


# Loss function
def vae_loss(y_true, y_pred):
    reconstruction_loss = K.mean(K.square(y_true - y_pred))
    kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    return reconstruction_loss + kl_loss


# In[61]:


# Compile the model
vae.compile(optimizer='adam', loss=vae_loss)


# In[63]:


# Fit the model
vae.fit(X_scaled, X_scaled, epochs=100, batch_size=32)# Print success message
print("VAE model training completed.")


# In[22]:


# Encode the data using the trained VAE
encoded_data = vae.predict(X_scaled)

# Print the encoded data shape
print("Encoded data shape:", encoded_data.shape)

