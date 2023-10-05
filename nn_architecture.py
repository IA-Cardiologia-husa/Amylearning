from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np

def conv2D_module(inputs, filters, kernel_size=3, padding="valid", pool_size=2):
	
	"""
	CONV => RELU => CONV => RELU => MAXPOOL
	"""
	
	x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
			   kernel_initializer='he_normal')(inputs)
	x = Activation("relu")(x)
	x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding,
			   kernel_initializer='he_normal')(inputs)
	x = Activation("relu")(x)


def UNet(input_size, depth, num_classes, filters, batch_norm):
	
	"""
	UNet (Ronneberger, 2015) implementation in tensorflow.keras
	using Keras Functional API.
	"""
	
	# Input layer
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	for layer in range(depth):
		x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
		
		if batch_norm: 
			x = BatchNormalization()(x)
			x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
			x_down = BatchNormalization()(x)
		else:
			x_down = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
		
		down_list.append(x_down)
		x = MaxPooling2D(pool_size=2)(x_down)
		filters = filters*2
	
	# Bottom
	x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
	if batch_norm: x = BatchNormalization()(x)
	x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
	if batch_norm: x = BatchNormalization()(x)
	
	# Decoding
	for layer in reversed(down_list):
		filters = filters // 2
		x = UpSampling2D((2,2))(x)
		x = concatenate([x, layer])
		x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
		if batch_norm: x = BatchNormalization()(x)
		x = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
		if batch_norm: x = BatchNormalization()(x)
	
	# Output layer
	x = Conv2D(filters=num_classes, kernel_size=1)(x)
	if batch_norm: x = BatchNormalization()(x)
	outputs = Activation("softmax")(x)
	
	model = Model(inputs, outputs)
	return model

def UNetAvg(input_size, depth, num_classes, filters, batch_norm):
	
	"""
	UNet (Ronneberger, 2015) implementation in tensorflow.keras
	using Keras Functional API.
	"""
	
	# Input layer
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	for layer in range(depth):
		x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
		
		if batch_norm: 
			x = BatchNormalization()(x)
			x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
			x_down = BatchNormalization()(x)
		else:
			x_down = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
		
		down_list.append(x_down)
		x = AveragePooling2D(pool_size=2)(x_down)
		filters = filters*2
	
	# Bottom
	x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
	if batch_norm: x = BatchNormalization()(x)
	x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
	if batch_norm: x = BatchNormalization()(x)
	
	# Decoding
	for layer in reversed(down_list):
		filters = filters // 2
		x = UpSampling2D((2,2))(x)
		x = concatenate([x, layer])
		x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
		if batch_norm: x = BatchNormalization()(x)
		x = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
		if batch_norm: x = BatchNormalization()(x)
	
	# Output layer
	x = Conv2D(filters=num_classes, kernel_size=1)(x)
	if batch_norm: x = BatchNormalization()(x)
	outputs = Activation("softmax")(x)
	
	model = Model(inputs, outputs)
	return model

def UNetEcho(input_size, depth, num_classes, filters, batch_norm):
	
	"""
	UNet (Ronneberger, 2015) implementation in tensorflow.keras
	using Keras Functional API.
	"""
	
	# Input layer
	inputs = Input(input_size)
	x = inputs
	y = inputs
	
	# Encoding
	down_list = []
	for layer in range(depth):
		x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
		
		if batch_norm: 
			x = BatchNormalization()(x)
			x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
			x_down = BatchNormalization()(x)
		else:
			x_down = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
		
		down_list.append(x_down)
		x = MaxPooling2D(pool_size=2)(x_down)
		y = AveragePooling2D(pool_size=2)(y)
		x = concatenate([x,y])
		
		filters = filters*2
	
	# Bottom
	x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
	if batch_norm: x = BatchNormalization()(x)
	x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
	if batch_norm: x = BatchNormalization()(x)
	
	# Decoding
	for layer in reversed(down_list):
		filters = filters // 2
		x = UpSampling2D((2,2))(x)
		x = concatenate([x, layer])
		x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
		if batch_norm: x = BatchNormalization()(x)
		x = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
		if batch_norm: x = BatchNormalization()(x)
	
	# Output layer
	x = Conv2D(filters=num_classes, kernel_size=1)(x)
	if batch_norm: x = BatchNormalization()(x)
	outputs = Activation("softmax")(x)
	
	model = Model(inputs, outputs)
	return model

def UNetCombined(input_size, depth, num_classes, filters, batch_norm):
	
	"""
	UNet (Ronneberger, 2015) implementation in tensorflow.keras
	using Keras Functional API.
	"""
	
	# Input layer
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	for layer in range(depth):
		x = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
		
		if batch_norm: 
			x = BatchNormalization()(x)
			x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
			x_down = BatchNormalization()(x)
		else:
			x_down = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
		
		down_list.append(x_down)
		y = MaxPooling2D(pool_size=2)(x_down)
		z = AveragePooling2D(pool_size=2)(x_down)
		x = concatenate([y,z])
		filters = filters*2
	
	# Bottom
	x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
	if batch_norm: x = BatchNormalization()(x)
	x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
	if batch_norm: x = BatchNormalization()(x)
	
	# Decoding
	for layer in reversed(down_list):
		filters = filters // 2
		x = UpSampling2D((2,2))(x)
		x = concatenate([x, layer])
		x = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
		if batch_norm: x = BatchNormalization()(x)
		x = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'he_normal')(x)
		if batch_norm: x = BatchNormalization()(x)
	
	# Output layer
	x = Conv2D(filters=num_classes, kernel_size=1)(x)
	if batch_norm: x = BatchNormalization()(x)
	outputs = Activation("softmax")(x)
	
	model = Model(inputs, outputs)
	return model


def HyperUNet(input_size, 
			  num_classes, 
			  depth, 
			  stack_size, 
			  filter_size, 
			  filters_base, 
			  filter_grow, 
			  depth_batch_norm, 
			  stack_batch_norm, 
			  stack_activation, 
			  depth_activation, 
			  upsampling_type, 
			  dropout_rate = 0, 
			  dropout = False, 
			  kernel_initializer = 'he_normal',
			  kernel_regularizer = None,
			  bias_regularizer = None):
	
	
	# Input layer
	
	filters = filters_base
	
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	
	for layer in range(depth):
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		down_list.append(x)
		x = MaxPooling2D(pool_size=2)(x)
		filters = filters*filter_grow
	
	# Bottom
	for stack_layer in range(stack_size-1):
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=stack_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		if stack_batch_norm: 
			x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
			
		
	
	# Decoding
	for layer in reversed(down_list):
		
		if(upsampling_type == 'upsampling'):
			x = UpSampling2D((2,2))(x)
		elif(upsampling_type == 'deconvolution'):
			x = Conv2DTranspose(filters, 2, strides = (2, 2), activation = None, kernel_initializer = 'he_normal')(x) 
		elif(upsampling_type == 'bilinear'):
			x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, np.array(x.get_shape().as_list()[-3:-1])*2, method='bilinear'))(x)
			
			
		filters = filters // filter_grow
		x = concatenate([x, layer])
		
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
	
	if dropout:
		x = SpatialDropout2D(dropout_rate)(x)
	
	# Output layer
	outputs = Conv2D(filters=num_classes, kernel_size=1, activation='softmax')(x)
	
	model = Model(inputs, outputs)
	return model

def HyperUNetPRelu(input_size, num_classes, depth, stack_size, filter_size, filters_base, filter_grow, depth_batch_norm, stack_batch_norm, upsampling_type, dropout_rate = 0, dropout = False):
	
	
	# Input layer
	
	filters = filters_base
	
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	
	for layer in range(depth):
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), filter_size, activation=None, padding='same', kernel_initializer='he_normal')(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), filter_size, activation=None, padding='same', kernel_initializer='he_normal')(x)
		x = tf.keras.layers.PReLU()(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		down_list.append(x)
		x = MaxPooling2D(pool_size=2)(x)
		filters = filters*filter_grow
	
	# Bottom
	for stack_layer in range(stack_size-1):
		x = Conv2D(int(filters), filter_size, activation=None, padding='same', kernel_initializer='he_normal')(x)
		if stack_batch_norm: 
			x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), filter_size, activation=None, padding='same', kernel_initializer='he_normal')(x)
		x = tf.keras.layers.PReLU()(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
			
		if dropout:
			x = Dropout(dropout_rate)(x)
	
	# Decoding
	for layer in reversed(down_list):
		
		if(upsampling_type == 'upsampling'):
			x = UpSampling2D((2,2))(x)
		elif(upsampling_type == 'deconvolution'):
			x = Conv2DTranspose(filters, 2, strides = (2, 2), activation = None, kernel_initializer = 'he_normal')(x) 
		elif(upsampling_type == 'bilinear'):
			x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, np.array(x.get_shape().as_list()[-3:-1])*2, method='bilinear'))(x)
			
			
		filters = filters // filter_grow
		x = concatenate([x, layer])
		
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), filter_size, activation=None, padding='same', kernel_initializer='he_normal')(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), filter_size, activation=None, padding='same', kernel_initializer='he_normal')(x)
		x = tf.keras.layers.PReLU()(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		
	# Output layer
	outputs = Conv2D(filters=num_classes, kernel_size=1, activation='softmax')(x)
	
	model = Model(inputs, outputs)
	return model

def HyperUNetEcho(input_size, num_classes, depth, stack_size, filter_size, filters_base, filter_grow, depth_batch_norm, stack_batch_norm, stack_activation, depth_activation, upsampling_type, dropout_rate = 0, dropout = False):
	
	
	# Input layer
	
	filters = filters_base
	
	inputs = Input(input_size)
	x = inputs
	y = inputs
	
	# Encoding
	down_list = []
	
	for layer in range(depth):
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), filter_size, activation=stack_activation, padding='same', kernel_initializer='he_normal')(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), filter_size, activation=depth_activation, padding='same', kernel_initializer='he_normal')(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		down_list.append(x)
		x = MaxPooling2D(pool_size=2)(x)
		y = AveragePooling2D(pool_size=2)(y)
		x = concatenate([x,y])
		filters = filters*filter_grow
	
	# Bottom
	for stack_layer in range(stack_size-1):
		x = Conv2D(int(filters), filter_size, activation=stack_activation, padding='same', kernel_initializer='he_normal')(x)
		if stack_batch_norm: 
			x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), filter_size, activation=depth_activation, padding='same', kernel_initializer='he_normal')(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
			
		if dropout:
			x = Dropout(dropout_rate)(x)
	
	# Decoding
	for layer in reversed(down_list):
		
		if(upsampling_type == 'upsampling'):
			x = UpSampling2D((2,2))(x)
		elif(upsampling_type == 'deconvolution'):
			x = Conv2DTranspose(filters, 2, strides = (2, 2), activation = None, kernel_initializer = 'he_normal')(x) 
		elif(upsampling_type == 'bilinear'):
			x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, np.array(x.get_shape().as_list()[-3:-1])*2, method='bilinear'))(x)
			
			
		filters = filters // filter_grow
		x = concatenate([x, layer])
		
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), filter_size, activation=stack_activation, padding='same', kernel_initializer='he_normal')(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), filter_size, activation=depth_activation, padding='same', kernel_initializer='he_normal')(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		
	# Output layer
	outputs = Conv2D(filters=num_classes, kernel_size=1, activation='softmax')(x)
	
	model = Model(inputs, outputs)
	return model
	

def HyperUNetGAN(input_size, 
			  num_classes, 
			  depth, 
			  stack_size, 
			  filter_size, 
			  filters_base, 
			  filter_grow, 
			  depth_batch_norm, 
			  stack_batch_norm, 
			  stack_activation, 
			  depth_activation, 
			  upsampling_type, 
			  dropout_rate = 0, 
			  dropout = False, 
			  kernel_initializer = 'he_normal',
			  kernel_regularizer = None,
			  bias_regularizer = None):
	
	
	# Input layer
	
	filters = filters_base
	
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	
	for layer in range(depth):
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		down_list.append(x)
		x = MaxPooling2D(pool_size=2)(x)
		filters = filters*filter_grow
	
	# Bottom
	for stack_layer in range(stack_size-1):
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=stack_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		if stack_batch_norm: 
			x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
			
		
	
	# Decoding
	for layer in reversed(down_list):
		
		if(upsampling_type == 'upsampling'):
			x = UpSampling2D((2,2))(x)
		elif(upsampling_type == 'deconvolution'):
			x = Conv2DTranspose(filters, 2, strides = (2, 2), activation = None, kernel_initializer = 'he_normal')(x) 
		elif(upsampling_type == 'bilinear'):
			x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, np.array(x.get_shape().as_list()[-3:-1])*2, method='bilinear'))(x)
			
			
		filters = filters // filter_grow
		x = concatenate([x, layer])
		
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
	
	if dropout:
		x = SpatialDropout2D(dropout_rate)(x)
	
	# Output layer
	outputs = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(x)
	
	model = Model(inputs, outputs)
	return model

def HyperUNetDepthConv(input_size, 
			  num_classes, 
			  depth, 
			  stack_size, 
			  filter_size, 
			  filters_base, 
			  filter_grow, 
			  depth_batch_norm, 
			  stack_batch_norm, 
			  stack_activation, 
			  depth_activation, 
			  upsampling_type, 
			  dropout_rate = 0, 
			  dropout = False, 
			  kernel_initializer = 'he_normal',
			  kernel_regularizer = None,
			  bias_regularizer = None):
	
	
	# Input layer
	
	filters = filters_base
	
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	
	for layer in range(depth):
		for stack_layer in range(stack_size-1):
			x = DepthwiseConv2D(filter_size, 
					   activation=None, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			x = Conv2D(int(filters), 
					   1, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = DepthwiseConv2D(filter_size, 
				   activation=None, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		x = Conv2D(int(filters), 
				   1, 
				   activation=stack_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		down_list.append(x)
		x = MaxPooling2D(pool_size=2)(x)
		filters = filters*filter_grow
	
	# Bottom
	for stack_layer in range(stack_size-1):
		x = DepthwiseConv2D(filter_size, 
				   activation=None, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		x = Conv2D(int(filters), 
				   1, 
				   activation=stack_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		if stack_batch_norm: 
			x = BatchNormalization()(x)
		
		x = DepthwiseConv2D(filter_size, 
				   activation=None, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		x = Conv2D(int(filters), 
				   1, 
				   activation=stack_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
			
		
	
	# Decoding
	for layer in reversed(down_list):
		
		if(upsampling_type == 'upsampling'):
			x = UpSampling2D((2,2))(x)
		elif(upsampling_type == 'deconvolution'):
			x = Conv2DTranspose(filters, 2, strides = (2, 2), activation = None, kernel_initializer = 'he_normal')(x) 
		elif(upsampling_type == 'bilinear'):
			x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, np.array(x.get_shape().as_list()[-3:-1])*2, method='bilinear'))(x)
			
			
		filters = filters // filter_grow
		x = concatenate([x, layer])
		
		for stack_layer in range(stack_size-1):
			x = DepthwiseConv2D(filter_size, 
					   activation=None, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			x = Conv2D(int(filters), 
					   1, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = DepthwiseConv2D(filter_size, 
				   activation=None, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		x = Conv2D(int(filters), 
				   1, 
				   activation=stack_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
	
	if dropout:
		x = SpatialDropout2D(dropout_rate)(x)
	
	# Output layer
	outputs = Conv2D(filters=num_classes, kernel_size=1, activation='softmax')(x)
	
	model = Model(inputs, outputs)
	return model
	
def HyperUNet2Masks(input_size,
			  depth, 
			  stack_size, 
			  filter_size, 
			  filters_base, 
			  filter_grow, 
			  depth_batch_norm, 
			  stack_batch_norm, 
			  stack_activation, 
			  depth_activation, 
			  upsampling_type, 
			  dropout_rate = 0, 
			  dropout = False, 
			  kernel_initializer = 'he_normal',
			  kernel_regularizer = None,
			  bias_regularizer = None):
	
	
	# Input layer
	
	filters = filters_base
	
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	
	for layer in range(depth):
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		down_list.append(x)
		x = MaxPooling2D(pool_size=2)(x)
		filters = filters*filter_grow
	
	# Bottom
	for stack_layer in range(stack_size-1):
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=stack_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		if stack_batch_norm: 
			x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
			
		
	
	# Decoding
	for layer in reversed(down_list):
		
		if(upsampling_type == 'upsampling'):
			x = UpSampling2D((2,2))(x)
		elif(upsampling_type == 'deconvolution'):
			x = Conv2DTranspose(filters, 2, strides = (2, 2), activation = None, kernel_initializer = 'he_normal')(x) 
		elif(upsampling_type == 'bilinear'):
			x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, np.array(x.get_shape().as_list()[-3:-1])*2, method='bilinear'))(x)
			
			
		filters = filters // filter_grow
		x = concatenate([x, layer])
		
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
	
	if dropout:
		x = SpatialDropout2D(dropout_rate)(x)
	
	# Output layer
	outputs = Conv2D(filters=2, kernel_size=1, activation='sigmoid')(x)
	
	model = Model(inputs, outputs)
	return model

def HyperUNetApex(input_size, 
			  depth, 
			  stack_size, 
			  filter_size, 
			  filters_base, 
			  filter_grow, 
			  depth_batch_norm, 
			  stack_batch_norm, 
			  stack_activation, 
			  depth_activation, 
			  upsampling_type, 
			  dropout_rate = 0, 
			  dropout = False, 
			  kernel_initializer = 'he_normal',
			  kernel_regularizer = None,
			  bias_regularizer = None):
	
	
	# Input layer
	
	filters = filters_base
	
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	
	for layer in range(depth):
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		down_list.append(x)
		x = MaxPooling2D(pool_size=2)(x)
		filters = filters*filter_grow
	
	# Bottom
	for stack_layer in range(stack_size-1):
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=stack_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		if stack_batch_norm: 
			x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
			
		
	
	# Decoding
	for layer in reversed(down_list):
		
		if(upsampling_type == 'upsampling'):
			x = UpSampling2D((2,2))(x)
		elif(upsampling_type == 'deconvolution'):
			x = Conv2DTranspose(filters, 2, strides = (2, 2), activation = None, kernel_initializer = 'he_normal')(x) 
		elif(upsampling_type == 'bilinear'):
			x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, np.array(x.get_shape().as_list()[-3:-1])*2, method='bilinear'))(x)
			
			
		filters = filters // filter_grow
		x = concatenate([x, layer])
		
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
	
	if dropout:
		x = SpatialDropout2D(dropout_rate)(x)
	
	# Output layer
	outputs = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(x)
	
	model = Model(inputs, outputs)
	return model

def HyperUNet_1mask(input_size, 
			  depth, 
			  stack_size, 
			  filter_size, 
			  filters_base, 
			  filter_grow, 
			  depth_batch_norm, 
			  stack_batch_norm, 
			  stack_activation, 
			  depth_activation, 
			  upsampling_type, 
			  dropout_rate = 0, 
			  dropout = False, 
			  kernel_initializer = 'he_normal',
			  kernel_regularizer = None,
			  bias_regularizer = None):
	
	
	# Input layer
	
	filters = filters_base
	
	inputs = Input(input_size)
	x = inputs
	
	# Encoding
	down_list = []
	
	for layer in range(depth):
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
		down_list.append(x)
		x = MaxPooling2D(pool_size=2)(x)
		filters = filters*filter_grow
	
	# Bottom
	for stack_layer in range(stack_size-1):
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=stack_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		if stack_batch_norm: 
			x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
			
		
	
	# Decoding
	for layer in reversed(down_list):
		
		if(upsampling_type == 'upsampling'):
			x = UpSampling2D((2,2))(x)
		elif(upsampling_type == 'deconvolution'):
			x = Conv2DTranspose(filters, 2, strides = (2, 2), activation = None, kernel_initializer = 'he_normal')(x) 
		elif(upsampling_type == 'bilinear'):
			x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, np.array(x.get_shape().as_list()[-3:-1])*2, method='bilinear'))(x)
			
			
		filters = filters // filter_grow
		x = concatenate([x, layer])
		
		for stack_layer in range(stack_size-1):
			x = Conv2D(int(filters), 
					   filter_size, 
					   activation=stack_activation, 
					   padding='same', 
					   kernel_initializer = kernel_initializer,
					   kernel_regularizer = kernel_regularizer, 
					   bias_regularizer = bias_regularizer)(x)
			if stack_batch_norm: 
				x = BatchNormalization()(x)
		
		x = Conv2D(int(filters), 
				   filter_size, 
				   activation=depth_activation, 
				   padding='same', 
				   kernel_initializer = kernel_initializer,
				   kernel_regularizer = kernel_regularizer, 
				   bias_regularizer = bias_regularizer)(x)
		
		if depth_batch_norm: 
			x = BatchNormalization()(x)
	
	if dropout:
		x = SpatialDropout2D(dropout_rate)(x)
	
	# Output layer
	outputs = Conv2D(filters=1, kernel_size=1, activation='sigmoid')(x)
	
	model = Model(inputs, outputs)
	return model
