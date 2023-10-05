import numpy as np
import skimage
import skimage.segmentation
import scipy
import scipy.spatial
import cv2
import tensorflow as tf

# def find_mitral_plane_center(pred):
# 	la_mask = (pred == 3)
# 	lv_mask = (pred == 1)
#
# 	fr_la = np.logical_and(la_mask, morphology.dilation(lv_mask))
# 	fr_lv = np.logical_and(lv_mask, morphology.dilation(la_mask))
# 	fr = np.logical_or(fr_la,fr_lv)
#
# 	(x, y) = scipy.ndimage.center_of_mass(fr)
# 	return (x,y)
#
# def find_apex(pred, x_mp, y_mp):
# 	lv_mask = (pred == 1)
# 	perimeter = np.logical_xor(lv_mask,morphology.erosion(lv_mask))
# 	coords = measure.regionprops(perimeter.astype(int))[0]['coords']
# 	max_dist2 = 0
# 	x_apex = x_mp
# 	y_apex = y_mp
# 	for (x,y) in coords:
# 		dist2  = (x_mp-x)**2+(y_mp-y)**2
# 		if(dist2 > max_dist2):
# 			max_dist2 = dist2
# 			x_apex = x
# 			y_apex = y
#
# 	return (x_apex,y_apex)

def find_disks(pred, x_mp, y_mp, x_apex, y_apex):
	lv_mask = (pred == 1)
	perimeter = np.logical_xor(lv_mask,skimage.morphology.erosion(lv_mask))

	#direction vector for long axis
	vx = (x_apex-x_mp)/20
	vy = (y_apex-y_mp)/20
	#direction vector for perpendicular axis
	ux = vy
	uy = -vx

	coord_list = []

	for i in range(20):
		x_0 = x_mp + vx/2 + vx*i
		y_0 = y_mp + vy/2 + vy*i

		yl_list, xl_list = skimage.draw.line(int(y_0), int(x_0), int(y_0-100*uy), int(x_0-100*ux))
		validxl = np.logical_and(xl_list>=0,xl_list<pred.shape[1])
		validyl = np.logical_and(yl_list>=0,yl_list<pred.shape[0])
		validl = np.logical_and(validxl, validyl)
		xl_list = xl_list[validl]
		yl_list = yl_list[validl]

		yr_list, xr_list = skimage.draw.line(int(y_0+100*uy), int(x_0+100*ux), int(y_0), int(x_0))
		validxr = np.logical_and(xr_list>=0,xr_list<pred.shape[1])
		validyr = np.logical_and(yr_list>=0,yr_list<pred.shape[0])
		validr = np.logical_and(validxr, validyr)
		xr_list = xr_list[validr]
		yr_list = yr_list[validr]

		line_drawingl = np.zeros(pred.shape)
		line_drawingr = np.zeros(pred.shape)
		line_drawingl[yl_list, xl_list] = 1
		line_drawingr[yr_list, xr_list] = 1


		coords = True
		labels = skimage.measure.label(np.logical_and(perimeter, line_drawingl), background=0, return_num=False, connectivity=2)
		if(labels.max() == 1):
			(y1, x1) = scipy.ndimage.center_of_mass(labels==1)
		else:
			labels = skimage.measure.label(np.logical_and(perimeter, skimage.morphology.dilation(line_drawingl)),
								   background=0, return_num=False, connectivity=2)
			if(labels.max() >= 1):
				(y1, x1) = scipy.ndimage.center_of_mass(labels==1)
			else:
				coords = False

		labels = skimage.measure.label(np.logical_and(perimeter, line_drawingr), background=0, return_num=False, connectivity=2)
		if(labels.max() == 1):
			(y2, x2) = scipy.ndimage.center_of_mass(labels==1)
		else:
			labels = skimage.measure.label(np.logical_and(perimeter, skimage.morphology.dilation(line_drawingr)),
								   background=0, return_num=False, connectivity=2)
			if(labels.max() >= 1):
				(y2, x2) = scipy.ndimage.center_of_mass(labels==1)
			else:
				coords = False

		if coords:
			coord_list.append((x1,y1,x2,y2))
		else:
			print(i, "We could not find two vertices")


	return coord_list

def find_myo_regions(pred, x_mp, y_mp, x_apex, y_apex):
	lv_mask = (pred == 1)
	myo_mask = (pred == 2)
	perimeter = np.logical_xor(lv_mask,skimage.morphology.erosion(lv_mask))


	#direction vector for long axis
	vx = (x_apex-x_mp)*2./5.
	vy = (y_apex-y_mp)*2./5.
	#direction vector for perpendicular axis
	ux = vy
	uy = -vx

	#mask from top third of the ventricle to the apex
	x_0 = x_mp + vx*1
	y_0 = y_mp + vy*1

	yl_list, xl_list = skimage.draw.line(int(y_0), int(x_0), int(y_0-100*uy), int(x_0-100*ux))
	validxl = np.logical_and(xl_list>=0,xl_list<pred.shape[1])
	validyl = np.logical_and(yl_list>=0,yl_list<pred.shape[0])
	validl = np.logical_and(validxl, validyl)
	xl_list = xl_list[validl]
	yl_list = yl_list[validl]

	yr_list, xr_list = skimage.draw.line(int(y_0+100*uy), int(x_0+100*ux), int(y_0), int(x_0))
	validxr = np.logical_and(xr_list>=0,xr_list<pred.shape[1])
	validyr = np.logical_and(yr_list>=0,yr_list<pred.shape[0])
	validr = np.logical_and(validxr, validyr)
	xr_list = xr_list[validr]
	yr_list = yr_list[validr]

	mask_top = np.zeros(pred.shape)
	mask_top[yl_list, xl_list] = 1
	mask_top[yr_list, xr_list] = 1

	mask_top = skimage.segmentation.flood_fill(mask_top, (0,np.rint(x_apex).astype(int)), 1, connectivity=0.5).astype(bool)

	#mask from bottom third of the ventricle to the valve
	x_0 = x_mp + vx*2
	y_0 = y_mp + vy*2

	yl_list, xl_list = skimage.draw.line(int(y_0),int(x_0), int(y_0-100*uy), int(x_0-100*ux))
	validxl = np.logical_and(xl_list>=0,xl_list<pred.shape[1])
	validyl = np.logical_and(yl_list>=0,yl_list<pred.shape[0])
	validl = np.logical_and(validxl, validyl)
	xl_list = xl_list[validl]
	yl_list = yl_list[validl]

	yr_list, xr_list = skimage.draw.line(int(y_0+100*uy),int(x_0+100*ux), int(y_0), int(x_0))
	validxr = np.logical_and(xr_list>=0,xr_list<pred.shape[1])
	validyr = np.logical_and(yr_list>=0,yr_list<pred.shape[0])
	validr = np.logical_and(validxr, validyr)
	xr_list = xr_list[validr]
	yr_list = yr_list[validr]

	mask_bottom = np.zeros(pred.shape)
	mask_bottom[yl_list, xl_list] = 1
	mask_bottom[yr_list, xr_list] = 1

	mask_bottom = skimage.segmentation.flood_fill(mask_bottom, (pred.shape[0]-1, np.rint(x_mp).astype(int)), 1, connectivity=1).astype(bool)

	#mask from the long axis to the left and to the right

	yl_list, xl_list = skimage.draw.line(int(y_mp), int(x_mp), int(y_mp-100*vy), int(x_mp-100*vx))
	validxl = np.logical_and(xl_list>=0,xl_list<pred.shape[1])
	validyl = np.logical_and(yl_list>=0,yl_list<pred.shape[0])
	validl = np.logical_and(validxl, validyl)
	xl_list = xl_list[validl]
	yl_list = yl_list[validl]

	yr_list, xr_list = skimage.draw.line(int(y_mp+100*vy), int(x_mp+100*vx), int(y_mp), int(x_mp))
	validxr = np.logical_and(xr_list>=0,xr_list<pred.shape[1])
	validyr = np.logical_and(yr_list>=0,yr_list<pred.shape[0])
	validr = np.logical_and(validxr, validyr)
	xr_list = xr_list[validr]
	yr_list = yr_list[validr]

	mask_lax = np.zeros(pred.shape)
	mask_lax[yl_list, xl_list] = 1
	mask_lax[yr_list, xr_list] = 1

	mask_left = skimage.segmentation.flood_fill(mask_lax, (np.rint(y_apex).astype(int),0), 1, connectivity=1).astype(bool)
	mask_right = ~mask_left

	myo_mask_1 = myo_mask & (~mask_top & mask_right)
	myo_mask_2 = myo_mask & (mask_top & mask_bottom & mask_right)
	myo_mask_3 = myo_mask & (~mask_bottom & mask_right)
	myo_mask_4 = myo_mask & (~mask_bottom & mask_left)
	myo_mask_5 = myo_mask & (mask_top & mask_bottom & mask_left)
	myo_mask_6 = myo_mask & (~mask_top & mask_left)

	return [myo_mask_1, myo_mask_2, myo_mask_3, myo_mask_4, myo_mask_5, myo_mask_6]

def estimate_myocardium_1mask(pred, x_mp1, y_mp1, x_mp2, y_mp2, myo_thickness = 50):

	size = 2*myo_thickness+1
	xx = np.tensordot(np.ones(size), np.arange(-myo_thickness, myo_thickness+1), axes=0)
	yy = np.tensordot(np.arange(-myo_thickness, myo_thickness+1), np.ones(size), axes=0)
	kernel = (xx**2+yy**2<myo_thickness**2).astype(np.uint8)
	pred_myo = cv2.dilate(pred.astype(np.uint8), kernel=kernel)
	# pred_myo = skimage.morphology.binary_dilation(pred, kernel)-pred
	pred_la = skimage.morphology.binary_dilation(pred)-pred
	# pred_myo = tf.raw_ops.Conv2D(input=pred[np.newaxis, ..., np.newaxis].astype(float),filter=kernel[..., np.newaxis, np.newaxis], strides=[1,1,1,1], padding = 'SAME')
	# pred_myo = (np.array(pred_myo) > 0)[0,...,0].astype(int) - pred
	pred_myo = pred_myo - pred

	ux = x_mp1-x_mp2
	uy = y_mp1-y_mp2

	yl_list, xl_list = skimage.draw.line(int(y_mp1), int(x_mp1), int(y_mp1-100*uy), int(x_mp1-100*ux))
	validxl = np.logical_and(xl_list>=0,xl_list<pred.shape[1])
	validyl = np.logical_and(yl_list>=0,yl_list<pred.shape[0])
	validl = np.logical_and(validxl, validyl)
	xl_list = xl_list[validl]
	yl_list = yl_list[validl]

	yr_list, xr_list = skimage.draw.line(int(y_mp1+100*uy), int(x_mp1+100*ux), int(y_mp1), int(x_mp1))
	validxr = np.logical_and(xr_list>=0,xr_list<pred.shape[1])
	validyr = np.logical_and(yr_list>=0,yr_list<pred.shape[0])
	validr = np.logical_and(validxr, validyr)
	xr_list = xr_list[validr]
	yr_list = yr_list[validr]

	mask_bottom = np.zeros(pred.shape)
	mask_bottom[yl_list, xl_list] = 1
	mask_bottom[yr_list, xr_list] = 1

	mask_bottom = skimage.segmentation.flood_fill(mask_bottom, (pred.shape[0]-1, np.rint(x_mp1).astype(int)), 1, connectivity=1).astype(int)

	pred_myo = pred_myo*(1-mask_bottom)
	pred_la = pred_la*mask_bottom

	pred = pred+2*pred_myo+3*pred_la

	return pred


def longitudinal_mask(pred, half_maxdepth=100):
	lv_mask = (pred==1) |(pred==3)
	perimeter = np.logical_xor(lv_mask, skimage.morphology.binary_erosion(lv_mask))

	size = 2*half_maxdepth+1
	xx = np.tensordot(np.ones(size), np.arange(-half_maxdepth, half_maxdepth+1), axes=0)
	yy = np.tensordot(np.arange(-half_maxdepth, half_maxdepth+1), np.ones(size), axes=0)

	kernel = np.zeros([*xx.shape,1,2])
	kernel[...,0,0] = xx / np.exp(np.log(np.sqrt(2))*np.sqrt(xx**2+yy**2))
	kernel[...,0,1] = yy / np.exp(np.log(np.sqrt(2))*np.sqrt(xx**2+yy**2))


	kernel[...,0,1] = kernel[...,0,1] * (xx**2+yy**2<half_maxdepth**2)
	kernel[...,0,0] = kernel[...,0,0] * (xx**2+yy**2<half_maxdepth**2)


	p_mask = tf.raw_ops.Conv2D(input=lv_mask[np.newaxis, ..., np.newaxis].astype(float),filter=kernel, strides=[1,1,1,1], padding = 'SAME')
	p_mask = p_mask.numpy()[0]

	normalization = np.sqrt(p_mask[...,0]**2+p_mask[...,1]**2)
	p_mask[...,0] = p_mask[...,0]/(normalization+1e-7)
	p_mask[...,1] = p_mask[...,1]/(normalization+1e-7)

	longitudinal = np.zeros(p_mask.shape)
	longitudinal[...,1] = p_mask[...,0]
	longitudinal[...,0] = -p_mask[...,1]

	return longitudinal

def find_mitral_plane_and_apex_1mask(lv_mask):
	perimeter = np.logical_xor(lv_mask,skimage.morphology.erosion(lv_mask))
	coords = skimage.measure.regionprops(perimeter.astype(int))[0]['coords']
	(y_cm, x_cm) = scipy.ndimage.center_of_mass(lv_mask)

	dist2_max = 0

	for (y0,x0) in coords:
		if (y0 < y_cm):
			dist2 = (x0-x_cm)**2+(y0-y_cm)**2
			if(dist2 > dist2_max):
				dist2_max = dist2
				x_ap = x0
				y_ap = y0

	def area_triangulo(ax,ay, bx,by,cx,cy):
		ux = bx - ax
		uy = by - ay
		vx = cx - ax
		vy = cy - ay

		cp = ux * vy - uy * vx

		return np.abs(cp)/2.

	def area_trapecio(ax,ay, bx,by,cx,cy, dx, dy):
		return scipy.spatial.ConvexHull([[ax,ay],[bx,by],[cx,cy],[dx,dy]]).volume


	max_dist1 = 0
	for (y1,x1) in coords:
		if (y1 > y_cm):
			dist = (x1-x_cm)**2+(y1-y_cm)**2
			if dist > max_dist1:
				max_dist1 = dist
				x_mp1 = x1
				y_mp1 = y1

	# max_area = 0
	# for (y2, x2) in coords:
	# 	area = area_triangulo(x_cm, y_cm, x_mp1, y_mp1, x2, y2)
	# 	if (y2 > y_cm):
	# 		if area > max_area:
	# 			max_area = area
	# 			x_mp2 = x2
	# 			y_mp2 = y2

	x_cm1 = np.where(lv_mask[int(y_cm),:])[0].min()
	x_cm2 = np.where(lv_mask[int(y_cm),:])[0].max()
	max_area = 0
	for (y2, x2) in coords:
		area = area_trapecio(x_cm1, y_cm, x_cm2, y_cm, x_mp1, y_mp1, x2, y2)
		if (y2 > y_cm):
			if area > max_area:
				max_area = area
				x_mp2 = x2
				y_mp2 = y2

	x_mpc = (x_mp1+x_mp2) / 2
	y_mpc = (y_mp1+y_mp2) / 2

	return (x_mpc,y_mpc), (x_mp1, y_mp1), (x_mp2, y_mp2), (x_ap, y_ap)

# def find_mitral_plane_and_apex_1mask_alternative(lv_mask):
# 	perimeter = np.logical_xor(lv_mask,skimage.morphology.erosion(lv_mask, selem = np.eye(3)))
# 	coords = skimage.measure.regionprops(perimeter.astype(int))[0]['coords']
# 	(y_cm, x_cm) = scipy.ndimage.center_of_mass(lv_mask)
#
# 	def area_triangulo(ax,ay, bx,by,cx,cy):
# 		ux = bx - ax
# 		uy = by - ay
# 		vx = cx - ax
# 		vy = cy - ay
#
# 		cp = ux * vy - uy * vx
#
# 		return np.abs(cp)/2.
#
# 	max_dist1 = 0
# 	for (y1,x1) in coords:
# 		if (y1 > y_cm):
# 			dist = (x1-x_cm)**2+(y1-y_cm)**2
# 			if dist > max_dist1:
# 				max_dist1 = dist
# 				x_mp1 = x1
# 				y_mp1 = y1
#
# 	max_area = 0
# 	for (y2, x2) in coords:
# 		area = area_triangulo(x_cm, y_cm, x_mp1, y_mp1, x2, y2)
# 		if (y2 > y_cm):
# 			if area > max_area:
# 				max_area = area
# 				x_mp2 = x2
# 				y_mp2 = y2
#
# 	x_mpc = (x_mp1+x_mp2) / 2
# 	y_mpc = (y_mp1+y_mp2) / 2
#
# 	xx = np.tensordot(np.ones(lv_mask.shape[0]), np.arange(lv_mask.shape[1]), axes = 0)
# 	yy = np.tensordot(np.arange(lv_mask.shape[0]), np.ones(lv_mask.shape[1]), axes = 0)
#
# 	N = lv_mask.sum()
# 	sum_x = (lv_mask*xx).sum()
# 	sum_x2 = (lv_mask*xx**2).sum()
# 	sum_y = (lv_mask*yy).sum()
# 	# sum_y2 = (lv_mask*yy**2).sum()
# 	sum_xy = (lv_mask*xx*yy).sum()
#
# 	a = (sum_xy + y_mpc*sum_x -x_mpc*sum_y - N * x_mpc * y_mpc) / (sum_x2-2*x_mpc*sum_x+N*x_mpc**2)
# 	b = y_mpc - a*x_mpc
#
# 	y_list, x_list = skimage.draw.line(np.rint(b).astype(int),0,np.rint(b+lv_mask.shape[1]*a).astype(int),lv_mask.shape[1])
# 	validx = np.logical_and(x_list>=0,x_list<lv_mask.shape[1])
# 	validy = np.logical_and(y_list>=0,y_list<lv_mask.shape[0])
# 	valid = np.logical_and(validx, validy)
# 	x_list = x_list[valid]
# 	y_list = y_list[valid]
#
# 	line_drawing = np.zeros(lv_mask.shape)
# 	line_drawing[y_list, x_list] = 1
#
# 	labels = skimage.measure.label(np.logical_and(perimeter, line_drawing), background=0, return_num=False, connectivity=2)
# 	# y_max = 0
# 	y_min = lv_mask.shape[0]
# 	if labels.max()==0:
# 		x_ap = x_mpc
# 		y_ap = y_mpc
#
# 		# raise ValueError(f'a = {a}, b={b}, ({x_mp1}, {y_mp1}) ({x_mp2}, {y_mp2}) ({x_mpc}, {y_mpc})')
# 	for l in range(1, labels.max()+1):
# 		(y, x) = scipy.ndimage.center_of_mass(labels==l)
# 		# if (y > y_max):
# 		# 	y_max = y
# 		# 	y_mpc = y
# 		# 	x_mpc = x
# 		if (y < y_min):
# 			y_min = y
# 			y_ap = y
# 			x_ap = x
#
# 	return (x_mpc,y_mpc), (x_mp1, y_mp1), (x_mp2, y_mp2), (x_ap, y_ap)



def find_disks_1mask(lv_mask, x_mp, y_mp, x_apex, y_apex):

	perimeter = np.logical_xor(lv_mask,skimage.morphology.erosion(lv_mask))

	#direction vector for long axis
	vx = (x_apex-x_mp)/20
	vy = (y_apex-y_mp)/20
	#direction vector for perpendicular axis
	ux = vy
	uy = -vx

	coord_list = []

	for i in range(20):
		x_0 = x_mp + vx/2 + vx*i
		y_0 = y_mp + vy/2 + vy*i

		yl_list, xl_list = skimage.draw.line(int(y_0),int(x_0), int(y_0-100*uy),int(x_0-100*ux))
		validxl = np.logical_and(xl_list>0,xl_list<lv_mask.shape[1])
		validyl = np.logical_and(yl_list>0,yl_list<lv_mask.shape[0])
		validl = np.logical_and(validxl, validyl)
		xl_list = xl_list[validl]
		yl_list = yl_list[validl]

		yr_list, xr_list = skimage.draw.line( int(y_0+100*uy),int(x_0+100*ux), int(y_0),int(x_0))
		validxr = np.logical_and(xr_list>0,xr_list<lv_mask.shape[1])
		validyr = np.logical_and(yr_list>0,yr_list<lv_mask.shape[0])
		validr = np.logical_and(validxr, validyr)
		xr_list = xr_list[validr]
		yr_list = yr_list[validr]


		line_drawing = np.zeros(lv_mask.shape)
		line_drawing[yl_list, xl_list] = 1
		line_drawing[yr_list, xr_list] = 1

		line_drawing = skimage.morphology.dilation(line_drawing)

		coords = True
		labels = skimage.measure.label(np.logical_and(perimeter, line_drawing), background=0, return_num=False, connectivity=2)
		if labels.max()<2:
			(x_1,y_1,x_2,y_2) = (x_0,y_0,x_0,y_0)
		else:
			intersections = []
			for label in range(1,labels.max()+1):
				(y, x) = scipy.ndimage.center_of_mass(labels==label)
				intersections.append([x,y])
			max_distance = 0
			for (x,y) in intersections:
				if ((x-x_0)**2+(y-y_0)**2) > max_distance:
					(x_1,y_1) = (x,y)
					max_distance = (x-x_0)**2+(y-y_0)**2
			max_distance = 0
			for (x,y) in intersections:
				if ((x-x_1)**2+(y-y_1)**2)> max_distance:
					(x_2,y_2) = (x,y)
					max_distance = (x-x_1)**2+(y-y_1)**2
			if (y_2 > y_1):
				(aux, auy) = (x_1, y_1)
				(x_1, y_1) = (x_2, y_2)
				(x_2, y_2) = (aux, auy)
		coord_list.append((x_1,y_1,x_2,y_2))

	return coord_list


def myo_regions_segmentation(pred):
	lv_mask = (pred==1)
	(x_mpc,y_mpc), (x_mp1, y_mp1), (x_mp2, y_mp2), (x_apex, y_apex) = find_mitral_plane_and_apex_1mask(lv_mask)
	region_list = find_myo_regions(pred, x_mpc, y_mpc, x_apex, y_apex)

	return region_list

def sort_vertices_anticlockwise(coords):
	yr = coords[:, 0]
	xc = coords[:, 1]
	center_xc = np.sum(xc)/xc.shape
	center_yr = np.sum(yr)/yr.shape
	theta = np.arctan2(yr-center_yr, xc-center_xc) * 180 / np.pi
	indices = np.argsort(theta)
	x = xc[indices]
	y = yr[indices]
	return np.array([y,x]).T


def check_intertwinned(dyas_list, sys_list, volumes):
	if (len(dyas_list)==0) or (len(sys_list)==0):
		return True, [volumes.argmax()], [volumes.argmin()]

	dyas_counter = 0
	sys_counter = 0

	if (dyas_list[0]<sys_list[0]):
		dyas = True
		dyas_counter = 1
		sys_counter = 0
	else:
		dyas = False
		dyas_counter = 0
		sys_counter = 1

	while (dyas_counter < len(dyas_list)) & (sys_counter < len(sys_list)):
		if(sys_list[sys_counter] < dyas_list[dyas_counter]):
			if(dyas == False):
				if(volumes[sys_list[sys_counter]]>volumes[sys_list[sys_counter-1]]):
					sys_list.pop(sys_counter)
				else:
					sys_list.pop(sys_counter-1)
				return False, dyas_list, sys_list
			dyas = False
			sys_counter+=1
		else:
			if(dyas == True):
				if(volumes[dyas_list[dyas_counter]]>volumes[dyas_list[dyas_counter-1]]):
					dyas_list.pop(dyas_counter-1)
				else:
					dyas_list.pop(dyas_counter)
				return False, dyas_list, sys_list
			dyas = True
			dyas_counter+=1
	if(dyas==False):
		if dyas_counter < (len(dyas_list)-1):
			last_dyas_counter = dyas_counter + volumes[dyas_list[dyas_counter:]].argmax()
			last_dyas = dyas_list[last_dyas_counter]
			dyas_list[dyas_counter] = last_dyas
			for i in range(dyas_counter+1,len(dyas_list)):
				dyas_list.pop()
	else:
		if sys_counter < (len(sys_list)-1):
			last_sys_counter = sys_counter + volumes[sys_list[sys_counter:]].argmin()
			last_sys = sys_list[last_sys_counter]
			sys_list[sys_counter] = last_sys
			for i in range(sys_counter+1,len(sys_list)):
				sys_list.pop()
	return True, dyas_list, sys_list

def find_peaks(x):
	x = np.array(x)
	peaks = list(1+np.where((x[1:-1] > np.minimum(x[:-2], x[2:])) & (x[1:-1] >= np.maximum(x[:-2], x[2:])))[0])
	if x[0]>x[1]:
		peaks.insert(0,0)
	if x[-1]>x[-2]:
		peaks.append(len(x)-1)

	return np.array(peaks)

def find_prominences(x):
	x = np.array(x)
	peaks = find_peaks(x)
	prominences = []
	max_onesided_prominence = []
	left_window = []
	right_window = []

	for peak in peaks:
		left_peaks = np.where(x[:peak]>x[peak])[0]
		right_peaks = peak + np.where(x[peak:]>x[peak])[0]

		if (len(left_peaks) > 0 ):
			left_peak = left_peaks.max()
			left_valley = x[left_peak:peak].min()
			left_prominence = x[peak]-left_valley
		else:
			left_peak = None
			left_valley = x[:(peak+1)].min()
			left_prominence = x[peak]-left_valley

		if (len(right_peaks) > 0 ):
			right_peak = right_peaks.min()
			right_valley = x[peak:right_peak].min()
			right_prominence = x[peak]-right_valley
		else:
			right_peak = None
			right_valley = x[peak:].min()
			right_prominence = x[peak]-right_valley

		if (left_peak is None):
			left_window.append(np.nan)
		else:
			left_window.append(left_peak)
		if (right_peak is None):
			right_window.append(np.nan)
		else:
			right_window.append(right_peak)

		prominences.append(min(left_prominence, right_prominence))
		max_onesided_prominence.append(max(left_prominence, right_prominence))
			
	return peaks, np.array(prominences), np.array(max_onesided_prominence), np.array(left_window), np.array(right_window)

def find_systole_dyastole(vol_curve):
	peaks, prominences, max_onesided_prominences, left_window, right_window = find_prominences(vol_curve)
	
	condition1 = (prominences > max_onesided_prominences.max()/4.)
	condition2 =  (np.isnan(left_window) | np.isnan(right_window)) & (max_onesided_prominences > 2*max_onesided_prominences.max()/3.)

	dyas_list = list(peaks[condition1 | condition2])

	peaks, prominences, max_onesided_prominences, left_window, right_window = find_prominences(-vol_curve)
	
	condition1 = (prominences > max_onesided_prominences.max()/4.)
	condition2 =  (np.isnan(left_window) | np.isnan(right_window)) & (max_onesided_prominences > 2*max_onesided_prominences.max()/3.)

	sys_list = list(peaks[condition1 | condition2])
	
	result = False
	while (result == False):
		result, dyas_list, sys_list = check_intertwinned(dyas_list, sys_list, vol_curve)

	return sys_list, dyas_list

# def find_systole_dyastole(vol_curve):
# 	peaks, prominences, max_prominences = find_prominences(vol_curve)
# 	dyas_list = list(peaks[prominences > prominences.max()/4.])
# 	if (prominences[0] < prominences.max()/4.) and (max_prominences[0] > 2*prominences.max()/3.):
# 		dyas_list.insert(0, peaks[0])
# 	if (prominences[-1] < prominences.max()/4.) and (max_prominences[-1] > 2*prominences.max()/3.):
# 		dyas_list.append(peaks[-1])
#
# 	peaks, prominences, max_prominences = find_prominences(-vol_curve)
# 	sys_list = list(peaks[prominences > prominences.max()/4.])
# 	if (prominences[0] < prominences.max()/4.) and (max_prominences[0] > 2*prominences.max()/3.):
# 		sys_list.insert(0, peaks[0])
# 	if (prominences[-1] < prominences.max()/4.) and (max_prominences[-1] > 2*prominences.max()/3.):
# 		sys_list.append(peaks[-1])
#
# 	result = False
# 	while (result == False):
# 		result, dyas_list, sys_list = check_intertwinned(dyas_list, sys_list, vol_curve)
#
# 	return sys_list, dyas_list

def old_find_systole_dyastole(smoothed_curve, reliable_frames):
	local_min = []
	local_max = []

	#We look for dyastole and systole frames
	index = np.where(reliable_frames)[0]
	for i in range(index.shape[0]):
		if (i==0):
			if (smoothed_curve[index[i+1]]>smoothed_curve[index[i]]):
				local_min.append(index[i])
			elif (smoothed_curve[index[i+1]]<smoothed_curve[index[i]]):
				local_max.append(index[i])
		elif (i==index.shape[0]-1):
			if (smoothed_curve[index[i-1]]<smoothed_curve[index[i]]):
				local_max.append(index[i])
			elif (smoothed_curve[index[i-1]]>smoothed_curve[index[i]]):
				local_min.append(index[i])
		else:
			if (smoothed_curve[index[i-1]]>smoothed_curve[index[i]]<smoothed_curve[index[i+1]]):
				local_min.append(index[i])
			elif (smoothed_curve[index[i-1]]<smoothed_curve[index[i]]>smoothed_curve[index[i+1]]):
				local_max.append(index[i])

	dyas_list = []
	sys_list = []
	i=0
	while (i <len(local_min)):
		j=i
		while ((j<len(local_min)-1) and ((local_min[j+1]-local_min[i])<3)):
			j+=1
		# range_min = max(0,local_min[i]-2)
		# range_max = min(shape[0], local_min[j]+2)
		# if(g["volume curve avg"][range_min:range_max].min()<(vol_min+0.4*(vol_max-vol_min))):
		# 	sys_list.append(range_min+g["volume curve avg"][range_min:range_max].argmin())
		search_range = (np.arange(shape[0])>=max(0,local_min[i]-2)) & (np.arange(shape[0])<min(shape[0], local_min[j]+2)) & reliable_frames
		if(g["volume curve avg"][np.where(search_range)].min()<(vol_min+0.4*(vol_max-vol_min))):
			sys_list.append(np.ma.masked_array(g["volume curve avg"][...],~search_range).argmin())
		i = j+1

	i=0
	while (i <len(local_max)):
		j=i
		while ((j<len(local_max)-1) and ((local_max[j+1]-local_max[i])<3)):
			j+=1
		# range_min = max(0,local_max[i]-2)
		# range_max = min(shape[0], local_max[j]+2)
		# if(g["volume curve avg"][range_min:range_max].max()>(vol_max-0.4*(vol_max-vol_min))):
		# 	dyas_list.append(range_min+g["volume curve avg"][range_min:range_max].argmax())
		search_range = (np.arange(shape[0])>=max(0,local_max[i]-2)) & (np.arange(shape[0])<min(shape[0], local_max[j]+2)) & reliable_frames
		if(g["volume curve avg"][np.where(search_range)].max()>(vol_max-0.4*(vol_max-vol_min))):
			dyas_list.append(np.ma.masked_array(g["volume curve avg"][...],~search_range).argmax())
		i = j+1




	result = False
	while (result == False):
		result, dyas_list, sys_list = check_intertwinned(dyas_list, sys_list, g['volume curve avg'][...])

	return sys_list, dyas_list
