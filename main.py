import pandas as pd
import math
import sys
import cv2
from numpy import array
from time import gmtime, strftime
from bigc import bigc
from index import alpha, beta, ari, dunn, relabeling, clusterMovements
from cvutils.io import imshow, imwrite
from utils import readBinaryImage
from fuzzy_kmodes import fKModes

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

	# FKMode
	avg_time1, avg_cost1, avg_accuracy1, db_indexes1, dunn_indexes1 = fKModes(dataX, [], [], 		len(cl_bigc), 20)
	avg_time2, avg_cost2, avg_accuracy2, db_indexes2, dunn_indexes2 = fKModes(dataX, [], cl_bigc, 	len(cl_bigc), 1)

	print ("Average time:\t\t",     avg_time1, avg_time2)
	print ("Average Cost:\t\t",     avg_cost1, avg_cost2)
	print ("Average Accuracy:\t",   avg_accuracy1, avg_accuracy2)
	print ("Average DB Index:\t",   sum(db_indexes1) / len(db_indexes1),    sum(db_indexes2) / len(db_indexes2))
	print ("Average Dunn Index:\t", sum(dunn_indexes1) / len(dunn_indexes1), sum(dunn_indexes2) / len(dunn_indexes2))
	#print ("Best DB Index:\t", min(db_indexes1), min(db_indexes2))
	#print ("DB Indexes:", db_indexes)
	#print ("Best Dunn Index:\t", max(dunn_indexes1), max(dunn_indexes2))
	#print ("Dunn Indexes:", dunn_indexes)

	print("\nInfo:: Ends at\t" + strftime("%Y-%m-%d %H:%M:%S", gmtime()))