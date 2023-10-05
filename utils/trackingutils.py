import numpy as np
from numba import jit
import tensorflow as tf
import scipy


# @jit(nopython=True)
# def squared_image_differences(image, old_image, region_halfsize=6, window_halfsize = 6):
# 	diff2 = np.zeros((image.shape[0],image.shape[1], 2*window_halfsize+1, 2*window_halfsize+1), dtype = 'uint16')
# 	sum_diff2 = np.zeros((image.shape[0],image.shape[1], 2*window_halfsize+1, 2*window_halfsize+1), dtype = 'float32')
#
# 	for i in range(2*window_halfsize+1):
# 		for j in range(2*window_halfsize+1):
# 			inc_x = i - window_halfsize
# 			inc_y = j - window_halfsize
#
# 			diff2[np.maximum(0,inc_x):(image.shape[0] + np.minimum(0, inc_x)),
# 				  np.maximum(0,inc_y):(image.shape[1] + np.minimum(0, inc_y)), i, j] =\
# 								(image[np.maximum(0,inc_x):(image.shape[0] + np.minimum(0, inc_x)),
# 									  np.maximum(0,inc_y):(image.shape[1] + np.minimum(0, inc_y))]-
# 								old_image[np.maximum(0,-inc_x):(image.shape[0] + np.minimum(0, -inc_x)),
# 									 np.maximum(0,-inc_y):(image.shape[1] + np.minimum(0, -inc_y))])**2
# 	for i in range(2*region_halfsize+1):
# 		for j in range(2*region_halfsize+1):
# 			inc_x = i - region_halfsize
# 			inc_y = j - region_halfsize
# 			sum_diff2[np.maximum(0,inc_x):(image.shape[0] + np.minimum(0, inc_x)),
# 					  np.maximum(0,inc_y):(image.shape[1] + np.minimum(0, inc_y)), :, :] +=\
# 							diff2[np.maximum(0,-inc_x):(image.shape[0] + np.minimum(0, -inc_x)),
# 									 np.maximum(0,-inc_y):(image.shape[1] + np.minimum(0, -inc_y)), :, :]
# 	return sum_diff2

@jit(nopython=True)
def squared_image_differences_bck(image, old_image, region_halfsize=6, window_halfsize = 6):
	sum_diff2 = np.zeros((image.shape[0],image.shape[1], 2*window_halfsize+1, 2*window_halfsize+1), dtype = 'float32')

	for x in range(image.shape[1]):
		for y in range(image.shape[0]):
			for ix in range(2*window_halfsize+1):
				for jy in range(2*window_halfsize+1):
					inc_x = ix - window_halfsize
					inc_y = jy - window_halfsize

					min_region_x = max(-region_halfsize, -(x-inc_x), -x)
					max_region_x = min(region_halfsize+1, image.shape[1]-(x-inc_x), image.shape[1]-x)
					min_region_y = max(-region_halfsize, -(y-inc_y), -y)
					max_region_y = min(region_halfsize+1, image.shape[0]-(y-inc_y), image.shape[0]-y)

					for rx in range(min_region_x, max_region_x):
						for ry in range(min_region_y, max_region_y):
							sum_diff2[y,x,jy,ix] += (old_image[y-inc_y+ry, x-inc_x+rx]-image[y+ry,x+rx])**2

	return sum_diff2

@jit(nopython=True)
def squared_image_differences_fwd(image, old_image, region_halfsize=6, window_halfsize = 6):
	sum_diff2 = np.zeros((image.shape[0],image.shape[1], 2*window_halfsize+1, 2*window_halfsize+1), dtype = 'float32')

	for x in range(image.shape[1]):
		for y in range(image.shape[0]):
			for ix in range(2*window_halfsize+1):
				for jy in range(2*window_halfsize+1):
					inc_x = ix - window_halfsize
					inc_y = jy - window_halfsize

					min_region_x = max(-region_halfsize, -x, -(x+inc_x))
					max_region_x = min(region_halfsize, image.shape[1]-x, image.shape[1]-(x+inc_x))
					min_region_y = max(-region_halfsize, -y, -(y+inc_y))
					max_region_y = min(region_halfsize, image.shape[0]-y, image.shape[0]-(y+inc_y))

					for rx in range(min_region_x, max_region_x):
						for ry in range(min_region_y, max_region_y):
							sum_diff2[y,x,jy,ix] += (old_image[y+ry,x+rx]-image[y+inc_y+ry,x+inc_x+rx])**2

	return sum_diff2

@jit(nopython=True)
def img_tracking(sum_diff2):
	window_halfsize = (sum_diff2.shape[-1]-1)//2
	image_tracking = np.zeros((sum_diff2.shape[0], sum_diff2.shape[1], 2))

	for x in range(sum_diff2.shape[1]):
		for y in range(sum_diff2.shape[0]):
			min_value = 1e300
			min_ix = 0
			min_jy = 0
			for ix in range(2*window_halfsize+1):
				for jy in range(2*window_halfsize+1):
					if sum_diff2[y,x,jy,ix] < min_value:
						min_ix = ix
						min_jy = jy
						min_value = sum_diff2[y,x,jy,ix]
			if(sum_diff2[y,x,window_halfsize,window_halfsize]==min_value):
				image_tracking[y,x,0] = window_halfsize
				image_tracking[y,x,1] = window_halfsize
			else:
				image_tracking[y,x,0] = min_ix
				image_tracking[y,x,1] = min_jy
	return image_tracking


@jit(nopython=True)
def img_tracking_subpixel(sum_diff2):
	window_halfsize = (sum_diff2.shape[-1]-1)//2
	image_tracking = np.zeros((sum_diff2.shape[0], sum_diff2.shape[1], 2))

	for x in range(sum_diff2.shape[1]):
		for y in range(sum_diff2.shape[0]):
			min_value = 1e300
			min_ix = 0
			min_jy = 0
			for ix in range(2*window_halfsize+1):
				for jy in range(2*window_halfsize+1):
					if sum_diff2[y,x,jy,ix] < min_value:
						min_ix = ix
						min_jy = jy
						min_value = sum_diff2[y,x,jy,ix]
			if(sum_diff2[y,x,window_halfsize,window_halfsize]==min_value):
				image_tracking[y,x,0] = window_halfsize
				image_tracking[y,x,1] = window_halfsize
			else:
				image_tracking[y,x,0] = min_ix
				image_tracking[y,x,1] = min_jy

	for x in range(sum_diff2.shape[1]):
		for y in range(sum_diff2.shape[0]):
			track_x = int(image_tracking[y,x,0])
			track_y = int(image_tracking[y,x,1])

			if((track_x != 0) and (track_x !=2*window_halfsize)):
				zm1 = sum_diff2[y,x,track_y,track_x-1]
				z0 = sum_diff2[y,x,track_y,track_x]
				zp1 = sum_diff2[y,x,track_y,track_x+1]

				a = zm1+zp1-2*z0
				b = (zp1-zm1)/2.
				c = z0

				if (a!=0.):
					x_sbpx = -b/(2*a)
				else:
					x_sbpx = 0.
			else:
				x_sbpx = 0.

			if((track_y != 0) and (track_y !=2*window_halfsize)):
				zm1 = sum_diff2[y,x,track_y-1,track_x]
				z0 = sum_diff2[y,x,track_y,track_x]
				zp1 = sum_diff2[y,x,track_y+1,track_x]

				a = zm1+zp1-2*z0
				b = (zp1-zm1)/2.
				c = z0

				if (a!=0.):
					y_sbpx = -b/(2*a)
				else:
					y_sbpx = 0.
			else:
				y_sbpx = 0.

			image_tracking[y,x,0] += x_sbpx
			image_tracking[y,x,1] += y_sbpx

	return image_tracking


#TODO: change tensorflow dependency to sth else?
#ANSWER: tensorflow seems to be way faster than other methods
def block_blur(image,window_halfsize, smoothing_halfsize,mask = None):
	image = image - window_halfsize

	size = 2*smoothing_halfsize+1
	xx = np.tensordot(np.ones(size), np.arange(-smoothing_halfsize, smoothing_halfsize+1), axes=0)
	yy = np.tensordot(np.arange(-smoothing_halfsize, smoothing_halfsize+1), np.ones(size), axes=0)
	kernel = (xx**2+yy**2<smoothing_halfsize**2).astype(np.float32)

	image2 = tf.raw_ops.Conv2D(input=image[np.newaxis, ...],filter=tf.tensordot(kernel, np.eye(2, dtype=('float32')), axes=0), strides=[1,1,1,1], padding = 'SAME')

	image2 = image2.numpy()

	if mask is not None:
		mask2 = tf.raw_ops.Conv2D(input=mask[np.newaxis,...,np.newaxis].astype(float),filter=tf.tensordot(kernel, np.eye(1, dtype=('float32')), axes=0), strides=[1,1,1,1], padding = 'SAME')
		mask2 = mask2.numpy()

		image2[0,...,0] = mask*image2[0,...,0]
		image2[0,...,1] = mask*image2[0,...,1]

		image2 = image2[0]/(mask2[0]+1e-7)
	else:
		image2 = image2[0]/np.sum(kernel)

	image2 = image2 + window_halfsize

	return image2

# @jit(nopython=True)
# def block_blur(image,window_halfsize, smoothing_halfsize):
# 	image = image - window_halfsize
#
# 	kernel = np.ones((2*smoothing_halfsize+1, 2*smoothing_halfsize+1), dtype='float32')
# 	kernel = kernel
#
# 	image2 = np.zeros_like(image, dtype='float32')
#
# 	for x in range(image.shape[1]):
# 		for y in range(image.shape[0]):
# 			min_inc_x = max(-x, -smoothing_halfsize)
# 			max_inc_x = min(image.shape[1]-x, smoothing_halfsize+1)
# 			min_inc_y = max(-y, -smoothing_halfsize)
# 			max_inc_y = min(image.shape[0]-y, smoothing_halfsize+1)
#
# 			for inc_x in range(min_inc_x, max_inc_x):
# 				for inc_y in range(min_inc_y, max_inc_y):
# 					image2[y,x,...]+= image[y+inc_y,x+inc_x,...]*kernel[inc_y+smoothing_halfsize,inc_x+smoothing_halfsize]
# 			image2[y,x,...] = image2[y,x,...]/((max_inc_x-min_inc_x)*(max_inc_y-min_inc_y))
# 	return image2

@jit(nopython=True)
def penalty_term(partial_solution_smoothed, window_halfsize):
	penalty = np.zeros((partial_solution_smoothed.shape[0], partial_solution_smoothed.shape[1],2*window_halfsize+1,2*window_halfsize+1))

	for ix in range(2*window_halfsize+1):
		for jy in range(2*window_halfsize+1):
			inc_x = ix - window_halfsize
			inc_y = jy - window_halfsize
			penalty[:,:, jy, ix] = (ix - partial_solution_smoothed[:,:,0])**2+(jy - partial_solution_smoothed[:,:,1])**2
	return penalty


def strain_dl(vmask, lmask, region_list):
	nabla_l_v = np.zeros_like(vmask)
	nabla_l_v[...,0] = lmask[...,0]*np.gradient(vmask, axis=1)[...,0] + lmask[...,1]*np.gradient(vmask, axis=0)[...,0]
	nabla_l_v[...,1] = lmask[...,0]*np.gradient(vmask, axis=1)[...,1] + lmask[...,1]*np.gradient(vmask, axis=0)[...,1]
	strain_field = lmask[...,0]*nabla_l_v[...,0]+lmask[...,1]*nabla_l_v[...,1]

	gls_rate = 0.
	rls_rate = np.zeros(6, dtype=float)

	area = 0
	for i in range(6):
		region = region_list[...,i]
		gls_rate += (region*strain_field).sum()
		rls_rate[i] = (region*strain_field).sum()/(region.sum()+1e-7)
		area += region_list[..., i].astype(int).sum()

	gls_rate = gls_rate/(area+1e-7)

	return strain_field, gls_rate, rls_rate

def gls_rate_speckle(old_x_gt, old_y_gt, new_x_gt, new_y_gt):
	old_total_length = 0
	new_total_length = 0

	old_regional_length = np.zeros(6)
	new_regional_length = np.zeros(6)

	for depth in range(5):
		old_x = old_x_gt[36*depth:36*(depth+1)]
		old_y = old_y_gt[36*depth:36*(depth+1)]

		new_x = new_x_gt[36*depth:36*(depth+1)]
		new_y = new_y_gt[36*depth:36*(depth+1)]

		old_segment_lenghts = np.sqrt((old_x[1:]-old_x[:-1])**2+(old_y[1:]-old_y[:-1])**2)
		new_segment_lenghts = np.sqrt((new_x[1:]-new_x[:-1])**2+(new_y[1:]-new_y[:-1])**2)

		old_total_length += old_segment_lenghts.sum()
		new_total_length += new_segment_lenghts.sum()

		for region in range(6):
			old_regional_length[region] += old_segment_lenghts[region*6:(region+1)*6].sum()
			new_regional_length[region] += new_segment_lenghts[region*6:(region+1)*6].sum()

	gls_rate = new_total_length/old_total_length -1.
	rls_rate = new_regional_length/old_regional_length -1.

	return gls_rate, rls_rate

def gls_rate_tracking(x_gt, y_gt, tracking, window_halfsize):
	old_total_length = 0
	new_total_length = 0

	old_regional_length = np.zeros(6)
	new_regional_length = np.zeros(6)

	fx =  scipy.interpolate.interp2d(np.arange(tracking.shape[1]),np.arange(tracking.shape[0]), tracking[:,:,0]-window_halfsize)
	fy =  scipy.interpolate.interp2d(np.arange(tracking.shape[1]),np.arange(tracking.shape[0]), tracking[:,:,1]-window_halfsize)

	inc_x_tr = []
	inc_y_tr = []

	for (x,y) in zip(x_gt,y_gt):
		inc_x_tr.append(fx(x,y)[0])
		inc_y_tr.append(fy(x,y)[0])

	inc_x_tr = np.array(inc_x_tr)
	inc_y_tr = np.array(inc_y_tr)

	for depth in range(5):

		x = x_gt[36*depth:36*(depth+1)]
		y = y_gt[36*depth:36*(depth+1)]

		inc_x = inc_x_tr[36*depth:36*(depth+1)]
		inc_y = inc_y_tr[36*depth:36*(depth+1)]

		old_segment_lengths = np.sqrt((x[1:]-x[:-1])**2+(y[1:]-y[:-1])**2)
		new_segment_lengths = np.sqrt((x[1:]+inc_x[1:]-x[:-1]-inc_x[:-1])**2+
									  (y[1:]+inc_y[1:]-y[:-1]-inc_y[:-1])**2)

		old_total_length += old_segment_lengths.sum()
		new_total_length += new_segment_lengths.sum()

		for region in range(6):
			old_regional_length[region] += old_segment_lengths[region*6:(region+1)*6].sum()
			new_regional_length[region] += new_segment_lengths[region*6:(region+1)*6].sum()

	gls_rate = new_total_length/old_total_length -1.
	rls_rate = new_regional_length/old_regional_length -1.

	return gls_rate, rls_rate
