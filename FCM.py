#!/usr/bin/python
import numpy as np
import random
import time
import copy
import csv

def loadCSV(fileName):
    fileHandler = open(fileName, "rt")
    lines = fileHandler.readlines()
    fileHandler.close()
    dataset = []
    for line in lines:
        instance = lineToTuple(line)
        dataset.append(instance)
    return dataset
	
def lineToTuple(line):
    ### remove leading/trailing white space and newlines
    cleanLine = line.strip()
    ### get rid of quotes
    cleanLine = cleanLine.replace('"', '')
    ### separate the fields
    lineList = cleanLine.split(",")
    ### convert strings into numbers
    stringsToNumbers(lineList)
    lineTuple = tuple(lineList)
    return lineTuple

def stringsToNumbers(myList):
    for i in range(len(myList)):
        if (isValidNumberString(myList[i])):
            myList[i] = float(myList[i])

def isValidNumberString(s):
  if len(s) == 0:
    return False
  if  len(s) > 1 and s[0] == "-":
      s = s[1:]
  for c in s:
    if c not in "0123456789.":
      return False
  return True
  
def calculateDistance(instance1, instance2):
    if instance1 == None or instance2 == None:
        return float("inf")
    sumOfSquares = 0
    if len(instance1) == len(instance2):
        for i in range(len(instance1)):
            sumOfSquares += (instance1[i] - instance2[i])**2
    else:
        print " instance length are not same!"
        print "instance 1 : " + str(instance1)
        print "instance 2 : " + str(instance2)
			
    return sumOfSquares
  
def findMeanInstance(instanceList):

    numInstances = len(instanceList)
    if (numInstances == 0):
        return
    numAttributes = len(instanceList[0])
    total_means = [0] * len(instanceList[0])
	
    for instance in instanceList:
        for i in range(numAttributes):
            total_means[i] += instance[i]
			
    for i in range(numAttributes):
        total_means[i] /= float(numInstances)
		
    return tuple(total_means)

def divide(x, y):

    try:
        result = x / y
    except ZeroDivisionError:
        if x !=0 or x!= 0.0:
            return "indefinite"
        else:
            return 1
    else:
        return result
		
def computeFuzzyW(membershipList, distance_Matrix_FuzzyW, m):

    W = list(distance_Matrix_FuzzyW)
	
    for i in range(len(membershipList)):
        for j in range(len(membershipList[0])):
            membershipList[i][j] = (membershipList[i][j])**m
			
    for i in range(len(membershipList)):
        for j in range(len(membershipList[0])):
			distance_Matrix_FuzzyW[i][j] *= membershipList[i][j]
			
	FuzzyW = 0
    for row in distance_Matrix_FuzzyW:
        for l in row:
            FuzzyW += l
			
    return W, FuzzyW
		
def createMembershipListWithoutDataset(membershipList, CentroidList):

    distance_Matrix = []
    for row in membershipList:
        instance = row[0]
        dist_row = []
        for centroid in CentroidList:
            dist = calculateDistance(instance, centroid)
            dist_row.append(dist)
        distance_Matrix.append(dist_row)
		
	uList_copy = np.array(membershipList)
    membershipList = uList_copy.T
    instanceList = membershipList[0]
    membershipList = np.delete(membershipList, (0), axis=0)
    membershipList = membershipList.T
	
    return membershipList, distance_Matrix
	
def computeFuzzyB(CentroidList, centroidGravity, membershipList, m):

    membershipList = np.array(membershipList)
    uListT = membershipList.T
	
    sumuij = []
    for columnuij in uListT:
        sumrow = 0
        for i in columnuij:
            sumrow += (i)**m
        sumuij.append(sumrow)
	
    DistanceB = []
    for centroid in CentroidList:
        dist = calculateDistance(centroid, centroidGravity)
        DistanceB.append(dist)
		
    B = list(DistanceB)		
		
    FuzzyB = 0
    for i in range(len(sumuij)):
        each_value = DistanceB[i] * sumuij[i]
        FuzzyB += each_value
		
    return B, FuzzyB
	
def computeFuzzyI(centroidGravity, membershipList, dataset, m):

    sumuij = 0
    for columnuij in membershipList:
        for i in columnuij:
            sumuij += (i)**m
			
    sumuij = []
    for columnuij in membershipList:
        sumrow = 0
        for i in columnuij:
            sumrow += (i)**m
        sumuij.append(sumrow)

    DistanceI = []			
    for instance in dataset:
        dist = calculateDistance(instance, centroidGravity)
        DistanceI.append(dist)
		
    I = list(DistanceI)	
		
    FuzzyI = 0
    for i in range(len(dataset)):
        FuzzyI += DistanceI[i] * sumuij[i]
		
    return I, FuzzyI
		
def centroidsFunction(dataset, i, membershipList):
    ### cj = sum(uij**m . xi)/sum(uij)**m
	
    copyinstances = list(dataset)
    instance = copyinstances[0]
    equationAbove = [0] * len(instance)
	
    equationBelow = 0
    for u in membershipList:
        instance = u[0]
        centroidValue = (u[i+1])**2
        equationBelow += centroidValue
        for x in range(len(instance)):
            equationAbove[x] += instance[x] * centroidValue
			
    newCentroid = [0] * len(instance)
    for j in range(len(instance)):
        newCentroid[j] = equationAbove[j]/equationBelow

    return newCentroid
	
def computeCentroids(dataset, CentroidList, membershipList):
    centroidsNewList = []
    for i in range(len(CentroidList)):
        newCentroid = centroidsFunction(dataset, i, membershipList)
        centroidsNewList.append(newCentroid)
		
    return centroidsNewList
  
def assign(instance, centroids, m):
    ### calculate each membership values for sample, i.e., uij
    ### uij	= 1/sum((xi-cj)/(xi-ck))**(2/m-1)
	
    each_uij = [0] * (len(centroids)+1)
    each_uij[0] = instance

    for i in range(len(centroids)):
        distAbove =  calculateDistance(instance, centroids[i])
        distBelow = 0
        sum_dist = 0
        sum_dist_list = []
        for j in range(len(centroids)):
            distBelow = calculateDistance(instance, centroids[j])
            result = divide(distAbove, distBelow)
            sum_dist_list.append(str(result))
        if "indefinite" in sum_dist_list:
            each_uij[i+1] = 0
        else:
            sum_dist_list = [x for x in sum_dist_list if x != "indefinite"]
            for t in sum_dist_list:
                sum_dist += float(t)**(1/float(m-1))
            each_uij[i+1] = 1/(sum_dist)
			
    return each_uij

def initializeMembershipMatrix(dataset, centroids, m):
    ### calculate membership values for a sample, i.e., each_uij value
    membershipList = []
    for instance in dataset:
        each_uij = assign(instance, centroids, m)
        membershipList.append(each_uij)
		
    return membershipList
  
def stopWithIteration(centroids, old_centroids, iterations):
    MAX_ITERATIONS = 1
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids
	
def stopWithThreshold(centroids, old_centroids, iterations, FuzzyW, oldFuzzyW):
    if iterations > 1:
        dif = FuzzyW - oldFuzzyW
        if FuzzyW == 0:
            diff_relative = 0
        else:
            diff_relative = dif/FuzzyW
        if abs(diff_relative) < 0.000001:
            return True
    else:
        return False
			
    return old_centroids == centroids
  
def runFuzzyCMeans(selectedCentroids, dataset, K, m, initCentroids=None):

    CentroidList = selectedCentroids
    oldCentroids = [[] for i in range(K)]
	
    ### STOPPING CRITERIA SETTINGS
    iterations = 0
    oldFuzzyW = 0
    gravity_center = findMeanInstance(dataset)
    for instance in dataset:
            oldFuzzyW += calculateDistance(instance,gravity_center)
    FuzzyW = 0
	
    #### LOOP FOR STOPPING CRITERIA WITH 2 OPTIONS: 
	#"stopWithIteration" is number of iteration, 
	#while not (stopWithIteration(CentroidList, oldCentroids, iterations)):
	
    #"stopWithThreshold" is relative difference for threshold value
    while not (stopWithThreshold(CentroidList, oldCentroids, iterations, FuzzyW, oldFuzzyW)):	
        iterations += 1
        #### CALCULATE MEMBERSHIP MATRIX 
        membershipList = initializeMembershipMatrix(dataset, CentroidList, m)
		
        #### STORE PREVIOUS CENTROID VALUES
        oldCentroids = list(CentroidList)
		
		#### CALCULATE NEW CENTROID VALUES
        CentroidList = computeCentroids(dataset, CentroidList, membershipList)
		
		#### STORE PREVIOUS FUZZYW VALUES for STOPING CRITERIA 
        oldFuzzyW = FuzzyW
		
        #### STORE MULTIPLE MEMBERSHIP MATRIX
        membershipListcopy = copy.copy(membershipList)
	
        ###	CALCULATE DISTANCE MATRIX AND MEMBERSHIP MATRIX WITHOUT INSTANCES 	
        membershipListNoDataset, distance_Matrix = createMembershipListWithoutDataset(membershipListcopy, CentroidList)
		
        #### STORE MULTIPLE MEMBERSHIP MATRIX WITHOUT INSTANCES COLUMN BECAUSE SAME MATRIX IS NEEDED TO USE AND UPDATE IN DIFFERENT COMPUTATIONS
        membershipList_FuzzyW = copy.copy(membershipListNoDataset)
        membershipList_FuzzyB = copy.copy(membershipListNoDataset)
        membershipList_FuzzyI = copy.copy(membershipListNoDataset)
		
        #### STORE MULTIPLE DISTANCE MATRIX
        distanceMatrix_FuzzyW = copy.copy(distance_Matrix)

	    #### CALCULATE FUZZY W
        W, FuzzyW = computeFuzzyW(membershipList_FuzzyW, distanceMatrix_FuzzyW, m)
		
        #### CALCULATE FUZZY B FOR GRAVITY CENTER OF ALL INSTANCES
        centroidGravity = findMeanInstance(dataset)
        B, FuzzyB = computeFuzzyB(CentroidList, centroidGravity, membershipList_FuzzyB, m)
		
        #### CALCULATE FUZZY I
        I, FuzzyI = computeFuzzyI(centroidGravity, membershipList_FuzzyI, dataset, m)
		
    return membershipList, FuzzyW, FuzzyB, FuzzyI, iterations
	

### LOAD CSV FILE: DATASET
dataset = loadCSV("TestData.csv")

### ASSIGN K VALUE (the number of clusters)
K=3
print "------------------------------------------------"
print "dataset is loaded!"

### SELECT RANDOM CENTROIDS
random.seed(time.time())
selectedCentroids = random.sample(dataset, K)
print "the number of cluster (K) is assigned: " + str(K)
ID=1
print "------------------------------------------------"
print "centroids are selected randomly: "
for centroid in selectedCentroids:
	print "centroid " + str(ID) + ": " + str(centroid)
	ID += 1
print "------------------------------------------------"

### ASSIGN M VALUE (the fuzziness coefficient by default, m = 2. If m = 1, clustering is crisp. If m > 1, clustering becomes fuzzy)
m=2
print "the fuzziness coefficient (m) is assigned: " + str(m)
print "------------------------------------------------"

### RUN Fuzzy C-MEANS method
print "Fuzzy C-Means is running..."
membershipList, FuzzyW, FuzzyB, FuzzyI, iterations = runFuzzyCMeans(selectedCentroids, dataset, K, m, True)
print "Fuzzy C-Means is finished"
print "------------------------------------------------"
print "The number of iterations: " + str(iterations)
print "------------------------------------------------"
print "The fuzzy within inertia (FUZZYW): " + str(FuzzyW)
print "The fuzzy between inertia (FUZZYB): " + str(FuzzyB)
print "The fuzzy inertia (FUZZYI): " + str(FuzzyI)

### WRITE IN CSV FILE
fileFCM = csv.writer(open("FuzzyCMeans-membershipMatrix.csv", "wb"))

### CREATE Header
Header = []
#Title.append("dataset") #add datasets to first column of membership matrix
for i in range(K):
    Header.append("column " + str(i+1))
	
### add Header in csv
fileFCM.writerow(Header)

for membershipValue in membershipList:
    membershipValue.pop(0) #delete datasets from first column of membership matrix
    fileFCM.writerow(membershipValue)
print "The membership matrix is created as the \"FuzzyCMeans-membershipMatrix.csv\" file"
