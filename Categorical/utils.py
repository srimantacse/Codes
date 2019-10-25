import math
import random
import sys
import numpy as np
import cv2
from numpy import array
from sklearn.metrics.pairwise import euclidean_distances
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.interpolate import spline
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.font_manager as font_manager
from sklearn.metrics import pairwise_distances
from categorical_distance import hamming_dist
import time


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 
	
def nCr(n,r):
	f = math.factorial

	if n == 1:
		return 1
	return f(n) / (f(r) * f(n-r))
	
def normDist(dist, dMax):
	return dist/(dMax + 1)
	
def inverseNormDist(dist, dMax):
	return 1 - normDist(dist, dMax)
	
def shapley(data, P=5):
	rowCount   = data.shape[0]
	dMax       = 0.0

	print("\n>>================= Shapley ================================")
	print("Info:: Shapley running with P: " + str(P) + " Data Shape: " + str(data.shape)) 
	shapley  = [0] * rowCount

	random.seed(time.time())
	randomDataIndex = random.sample(range(0, rowCount), P)

	# Find dMax, distance
	print("Info:: Distnace Calculation Started...")

	selectedData = [data[x] for x in randomDataIndex]
	distance     = euclidean_distances(data, selectedData)
	dMax         = np.amax(distance)
	print("Info:: Calculatd Distance Shape: " + str(array(distance).shape))
	print("Info:: Calculatd Distance Max: "   + "{:0,.2f}".format(dMax))	

	# Find Shapley
	print("Info:: Shapley Calculation Started...")
	for i in range(rowCount):
		result = 0.0
		for j in range(len(randomDataIndex)):
			result = result + inverseNormDist(distance[i][j], dMax)
			
		shapley[i] = (result / 2.0)		
		
	print("Info:: Shapley Calculation Done.")
	print("-------------------------------------------------------------")

	return list(shapley), dMax

def density(data, radius=0.0, P=1, verbose=False):
	density  	 = []
	dMax     	 = 0.0
	rowCount 	 = data.shape[0]	
	randomData   = random.sample(range(0, rowCount), P)
	
	print("\n>>================= Density =================================")
	print("Info:: Density Calculation Started...")
	
	selectedData = [data.loc[x] for x in randomData]
	distance     = array(euclidean_distances(data, selectedData))
	distance     = distance / np.amax(distance)
	
	if radius <= 0.0:
		rad = np.mean(distance)
	else:
		rad = radius
	
	if verbose:
		it = 0
		logInterval = 1
		if rowCount > 100:
			logInterval = rowCount / 10

	for i in range(rowCount):
		# Calculate density of each point
		validPoint = 0
		for j in range(len(randomData)):
			if distance[i][j] < rad:
				validPoint = validPoint + 1
	
		density.append(validPoint)
		
		if dMax <= validPoint:
			dMax = validPoint

		# Log the interval count
		if verbose:
			it = it + 1
			if it % logInterval == 0:
				print("--Density Calculation " + str(int(it/ logInterval) * 10) +" %")

	print("Info:: Density Calculation Done.")
	print("-------------------------------------------------------------")
	return list(density), dMax

def regionQuery(data, pointIndex, eps):
	"""
	Find all points in dataset `data` within distance `eps` of point `P[pointIndex]`.

	This function calculates the distance between a point P[pointIndex] and every other 
	point in the dataset, and then returns only those points which are within a
	threshold distance `eps`.
	"""
	neighbors = []
	rowCount  = data.shape[0]

	for Pn in range(0, rowCount):
		# If the distance is below the threshold, add it to the neighbors list.
		if np.linalg.norm(data.loc[pointIndex] - data.loc[Pn]) < eps:
		   neighbors.append(Pn)
			
	return neighbors

def list_max(l):
	if len(array(l).shape) == 2:
		print("list_max: 2D List...")
		max_idx_y, max_val = list_max(l[0])
		max_idx_x = 0
		itr = 0
		for row in l:
			idx, val = list_max(row)
			if max_val < val:
				max_val = val
				max_idx_x = itr
				max_idx_y = idx
				
			itr = itr + 1
			
		return ([max_idx_x, max_idx_y], max_val)
	else:
		max_idx = np.argmax(l)
		max_val = l[max_idx]
		return (max_idx, max_val)
	
def list_min(l):
	if len(array(l).shape) == 2:
		print("list_min: 2D List...")
		min_idx_y, min_val = list_min(l[0])
		min_idx_x = 0
		itr = 0
		for row in l:
			idx, val = list_min(row)
			if min_val > val:
				min_val = val
				min_idx_x = itr
				min_idx_y = idx
				
			itr = itr + 1
			
		return ([min_idx_x, min_idx_y], min_val)
	else:
		min_idx = np.argmin(l)
		min_val = l[min_idx]
		return (min_idx, min_val)
	
def d(d1, d2):
	x = 0.0
	for i in range(d1.shape[0]):
		x = x + ((float(d1[i]) - float(d2[i]))**2)
	dist = math.sqrt(x)
	return dist
	
def indexOf(array, data, validIndex):
	for i in range(array.shape[0]):
		if i not in validIndex:
			continue
		if d(array.loc[i], data) == 0:
			return i
	
	print("#### Wrong index found!!!")
	return -1
	
def multiIndexOf(lst, item):
	return [i for i, x in enumerate(lst) if x == item]
	
def unique(listSet): 
    unique_list = (list(set(listSet))) 
    return unique_list
	
def readBinaryImage(filePath):
	bytes_read = open(filePath, "rb")
	a = np.fromfile(bytes_read, dtype=np.ubyte)
	return a
	
def isGrayScale(image):
	if(len(image.shape) < 3):
		return True
	elif len(image.shape)==3:
		return False
	else:
		print('Something Wrong in the Image!!!!')
		return False
		
def plotGraph(xAxisData, yAxisDataList, labelInfoList, smoothingRequired=True, saveTheImage=False, xAxisName='X axis', yAxisName='Y axis', frameName='Image',outputFileName='output'):
	print ('Ploting!!!!')
	
	csfont = {'fontname':'BELL MT'}
	style.use('ggplot')
	
	plt.title(frameName, **csfont)
	plt.ylabel(yAxisName, **csfont)
	plt.xlabel(xAxisName, **csfont)
	_lineStyle = ':'
	
	iteration = 0
	for yAxisData in yAxisDataList:
		if smoothingRequired and len(xAxisData) >= 4:
			print ('Smoothing applied.')
			T     = np.array(xAxisData)
			power = np.array(yAxisData)
			xnew  = np.linspace(T.min(), T.max(), 300)
			spl   = make_interp_spline(T, power, k=3) #BSpline object
			ynew  = spl(xnew)

			plt.plot(xnew, ynew, label=labelInfoList[iteration], linestyle = _lineStyle)
		else:
			plt.plot(xAxisData, yAxisData, label=labelInfoList[iteration], linestyle = _lineStyle)
			
		iteration = iteration + 1

	font = font_manager.FontProperties(family='BELL MT', weight='bold', style='normal',)
	plt.legend(loc='best', prop=font)
	
	if saveTheImage:
		fileExtension = '.png'
		print ('Image is saved.')
		plt.savefig(outputFileName + fileExtension)
	else:
		plt.show()
		
def plotBox(xAxisData, labels, xAxisName='X axis', yAxisName='Y axis'):
	csfont = {'fontname':'BELL MT'}
	plt.title('Box Plot', **csfont)

	plt.xlabel(xAxisName)
	plt.ylabel(yAxisName)
	
	bp = plt.boxplot(xAxisData, notch=True, vert=True, patch_artist=True, labels=labels)
	for box in bp['boxes']:
		box.set(color='black', linewidth=1)
		box.set(facecolor = 'lightblue' )
		box.set(hatch = '/')
	
	font = font_manager.FontProperties(family='BELL MT', weight='bold', style='normal')
	plt.legend(loc='best', prop=font)
	plt.show()
	
def plotSeparateBox(data, labelList, xAxisName='X axis', yAxisName='Y axis'):
	csfont = {'fontname':'BELL MT'}
	
	colorSet = ['orchid', 'lightblue', 'pink', 'lightgreen', 'lightgray', 'firebrick', 'indigo']
	dataLength = len(data)
	
	idx, maxVal = list_max(data)
	print(idx, maxVal)
	idx, minVal = list_min(data)
	print(idx, minVal)
	
	extraRange = (maxVal - minVal) * .2
	print("Data Range: " + str(minVal) + "-" + str(maxVal))

	"""
	noOfCols = 3
	div = dataLength % 3
	noOfRows  = (dataLength / 3)
	if div != 0:
		noOfRows = noOfRows + 1
	noOfRows = int(noOfRows)
	"""
	noOfRows = 1
	noOfCols = dataLength
	print(noOfRows, noOfCols)
	
	fig, axes = plt.subplots(nrows=noOfRows, ncols=noOfCols, figsize=(8, 4))
	fig.suptitle('Box Plot', **csfont)
	iteration = 0
	breakLoop = False
	
	for i in range(0, noOfRows):
		if breakLoop:
			break
			
		for j in range(0, noOfCols):
			if iteration >= dataLength:
				breakLoop = True
				break
				
			if noOfRows > 1:
				_ax = axes[i, j]
			else:
				_ax = axes[j]
			bp = _ax.boxplot(data[iteration], notch=True, vert=True, patch_artist=True, labels=array(labelList[iteration]).reshape(1,))
			_ax.set_ylim(minVal - extraRange, maxVal + extraRange)
			iteration = iteration + 1
			
			_ax.set_ylabel(yAxisName)

			#_ax.xlabel(xAxisName)
			#_ax.ylabel(yAxisName)
			
			for box in bp['boxes']:
				box.set(color='black', linewidth=1)
				box.set(facecolor = colorSet[iteration % len(colorSet)] )
				box.set(hatch = '/')
		
	plt.subplots_adjust(bottom=0.15, wspace=1)
	plt.show()
	
def plotMultiImage(output_list):
    output_list_len = len(output_list)
    figure = plt.figure()
    for i in range(output_list_len):
        current_dict = output_list[i]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, output_list_len, i+1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap = plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color = "black")
            current_plot.set_xlim([0,260])
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)            
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list,rotation = 90)

    plt.show()
	
def ordered_dissimilarity_matrix(X):
    """The ordered dissimilarity matrix is used by visual assesement of tendency. It is a just a a reordering 
    of the dissimilarity matrix.
    Parameters
    ----------
    X : matrix
        numpy array
    Return
    -------
    ODM : matrix
        the ordered dissimalarity matrix .
    """

    # Step 1 :

    I = []

    R = pairwise_distances(X)
    P = np.zeros(R.shape[0], dtype="int")

    argmax = np.argmax(R)

    j = argmax % R.shape[1]
    i = argmax // R.shape[1]

    P[0] = i
    I.append(i)

    K = np.linspace(0, R.shape[0] - 1, R.shape[0], dtype="int")
    J = np.delete(K, i)

    # Step 2 :

    for r in range(1, R.shape[0]):

        p, q = (-1, -1)

        mini = np.max(R)

        for candidate_p in I:
            for candidate_j in J:
                if R[candidate_p, candidate_j] < mini:
                    p = candidate_p
                    q = candidate_j
                    mini = R[p, q]

        P[r] = q
        I.append(q)

        ind_q = np.where(np.array(J) == q)[0][0]
        J = np.delete(J, ind_q)

    # Step 3

    ODM = np.zeros(R.shape)

    for i in range(ODM.shape[0]):
        for j in range(ODM.shape[1]):
            ODM[i, j] = R[P[i], P[j]]

    # Step 4 :

    return ODM
	
def vat_cluster(X):
	"""VAT means Visual assesement of tendency. basically, it allow to asses cluster tendency
	through a map based on the dissimiliraty matrix. 
	Parameters
	----------
	X : matrix
		numpy array
	Return
	-------
	ODM : matrix
		the ordered dissimalarity matrix plotted.
	"""

	# Step 1 :

	I = []
	
	R = pairwise_distances(X)
	P = np.zeros(R.shape[0], dtype="int")

	argmax = np.argmax(R)

	j = argmax % R.shape[1]
	i = argmax // R.shape[1]

	P[0] = i
	I.append(i)

	K = np.linspace(0, R.shape[0] - 1, R.shape[0], dtype="int")
	J = np.delete(K, i)

	# Step 2 :

	for r in range(1, R.shape[0]):

		p, q = (-1, -1)

		mini = np.max(R)

		for candidate_p in I:
			for candidate_j in J:
				if R[candidate_p, candidate_j] < mini:
					p = candidate_p
					q = candidate_j
					mini = R[p, q]

		P[r] = q
		I.append(q)

		ind_q = np.where(np.array(J) == q)[0][0]
		J = np.delete(J, ind_q)

	# Step 3

	ODM = np.zeros(R.shape)

	for i in range(ODM.shape[0]):
		for j in range(ODM.shape[1]):
			ODM[i, j] = R[P[i], P[j]]

	ODM = ODM.astype('int32') 
	ODM = np.interp(ODM, (ODM.min(), ODM.max()), (0, 255))
	ODM = ODM.astype('uint8')
	return (ODM)


