import pandas as pd
import math
import sys
import cv2
from numpy import array
from time import gmtime, strftime
from bigc import bigc
from cvutils.io import imshow, imwrite
from fuzzy_kmedoids import fKMedoids
from kmedoids import KMedoids

def printFish():
	print("\n")
	print("           \/^*._         _")
	print("      .-*'`    `*-.._.-'/")
	print("    < * )) ... S.Kundu ...( ")
	print("      `*-._`._(__.--*\"`. _")
	print("            /\*            ")
	print("\n")


if __name__ == "__main__":
	printFish()
	print("Info:: Started at\t" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))	

	############################# Data Set Parameters #############################
	# 0: DONOT drop  | 1: drop Last      | 2: drop 	First		| 3: drop First & Last 		| 4: Special Case, First two
	# 0: No Target   | 1: Target at Last | 2: Target at First	| 3: Id First & Target Last | 4: Id First & Target second
	dataSetArray = [
					# name   			noOfPoint   dropInfo  	noOfClass	P			index	opt-delta
					['mushroom', 		8124,  		0, 			2, 			1000], 		#0
					]

	dataIndex = 0
	dataSet = 'data/'+ dataSetArray[dataIndex][0]
	P       = dataSetArray[dataIndex][4]
	drop    = dataSetArray[dataIndex][2]
	delta   = .7
	
	###############################################################################
	
	separator = '\s+'
	dataX = pd.read_csv(dataSet, sep=separator, header=None)
	if drop == 1:
		dataX = dataX.iloc[:, :-1]
	elif drop == 2:
		dataX = dataX.iloc[:, 1:]
	elif drop == 3:
		dataX = dataX.iloc[:, 1:-1]
	elif drop == 4:
		dataX = dataX.iloc[:, 2:]
		
	print("Info:: Original Data  Size: " + str(dataX.shape))
	
	# BIGC
	cl_bigc = bigc(dataX, delta, P)

	# KMedoid    and FKMedoid
	# GT-KMedoid and GT-FKMedoid

	print("\nInfo:: Ends at\t" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))