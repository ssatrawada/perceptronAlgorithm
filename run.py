import matplotlib 
import matplotlib.pyplot as plt 
import os
import numpy as np



def main():
	
	DIM = (28,28) #these are the dimensions of the image

	def load_image_files(n, path="images/"):

	    images = []
	    for f in os.listdir(os.path.join(path,str(n))): # read files in the path
	        p = os.path.join(path,str(n),f)
	        if os.path.isfile(p):
	            i = np.loadtxt(p)
	            assert i.shape == DIM # just check the dimensions here
	            # i is loaded as a matrix, but the command below flattens it into a vector
	            images.append(i.flatten())
	    return images# Load up these image files
	A = load_image_files(0)
	B = load_image_files(1)
	C2 = load_image_files(2)
	D3 = load_image_files(3)
	E4 = load_image_files(4)
	F5 = load_image_files(5)
	G6 = load_image_files(6)
	H7 = load_image_files(7)
	I8 = load_image_files(8)
	J9 = load_image_files(9)

	N = len(A[0]) # the total size
	assert N == DIM[0]*DIM[1] # just check our sizes to be sure
	# set up some random initial weights

	def act_func(val):
	    if val < 0: 
	        return 0
	    else: 
	        return 1

     #threshold = 0 
	def perceptron(image, weights): 
	    image0 = image
	    y = act_func(np.dot(image0, weights))
	    if y == 0: 
	        return 0
	    if y == 1: 
	        return 1

	accuracy = np.array([])
	numImages = 0
	numOf25blocks = 100
	for i in range(0, numOf25blocks):
	    numImages +=25
	    weights = np.random.normal(0,1,size=N)
	    correctCount = 0 
	    for i in range(0, numImages):
	        whichImage = np.random.choice([0,1], p = np.array([0.5, 0.5]))
	        if whichImage ==0: 
	            currImage = A[np.random.choice(len(A))]
	        else: 
	            currImage = B[np.random.choice(len(B))]
	        y = perceptron(currImage, weights)
	        if y ==0 and whichImage==1: 
	            weights = weights + currImage 
	        if y ==1 and whichImage ==0: 
	            weights = weights - currImage
	        else: 
	            correctCount +=1 
	    accuracy = np.append(accuracy, correctCount/numImages)
	plt.plot(np.array(range(0, numOf25blocks)), accuracy)
	plt.xticks(np.arange(0, numOf25blocks, step=10))
	plt.title("Average Accuracy on Blocks of 25 items")
	plt.xlabel("Number of increments of 25 to Total Number of Blocks")
	plt.ylabel("Accuracy")
	plt.show()

	#sdfjkl

	reshape_weight = np.reshape(weights, (28, 28))
	plt.imshow(reshape_weight)
	plt.title("Weight as an Image")
	plt.show()
	#code from above put in a method for convinence 
	imageArr = np.array([A, B, C2, D3, E4, F5, G6, H7, I8, J9 ])
	def train_general(numImages, one, two, imageArr):
	    weights = np.random.normal(0,1,size=N)
	    for i in range(0, numImages):
	        whichImage = np.random.choice([one,two], p = np.array([0.5, 0.5]))
	        if whichImage ==one: 
	            currImage = (imageArr[one])[np.random.choice(len(imageArr[one]))]
	        else: 
	            currImage = (imageArr[two])[np.random.choice(len(imageArr[two]))]
	        y = perceptron(currImage, weights)
	        if y ==0 and whichImage==two: 
	            weights = weights + currImage 
	        if y ==1 and whichImage ==one: 
	            weights = weights - currImage
	    return weights
	def numaccuracies_general(weights, numclass, i, j, imageArr):
	    countright = 0
	    for x in range(0, numclass):
	        whichImage = np.random.choice([i,j], p = np.array([0.5, 0.5]))
	        if whichImage ==i: 
	            currImage = imageArr[i][np.random.choice(len((imageArr[i])))]
	        else: 
	            currImage = imageArr[j][np.random.choice(len((imageArr[j])))]
	        y = perceptron(currImage, weights)
	        if y ==0 and whichImage==i: 
	            countright +=1
	        if y ==1 and whichImage ==j: 
	            countright +=1
	    return countright
	
	weights = [-4]
	weights = train_general(2000, 0, 1, imageArr)
	accuracy = np.array([])
	for i in np.arange(10,790,10):
	    zeroindex = (np.argpartition(abs(weights), i))[:i]
	    zero = np.zeros(i)
	    weights[zeroindex] = zero
	    accuracy = np.append(accuracy, numaccuracies_general(weights, 1000, 0, 1, imageArr)/1000)
	plt.plot(np.arange(10,790,10), accuracy)
	plt.title("Accuracy as Increasing Increments of 10 Weights are Zeroed")
	plt.ylabel("Accuracy")
	plt.xlabel("Number of Weights Zeroed")

	mat= np.zeros((10, 10))
	for i in range(0, 10):
	    row = np.array([])
	    for j in range(0, i):
	        weights = train_general(2500, i, j, imageArr)
	        allaccuracy = np.array([])
	        for x in range(0, 100):
	            allaccuracy = np.append(allaccuracy, numaccuracies_general(weights, 1000,i, j, imageArr)/1000)
	        mat[i][j] = np.mean(allaccuracy)
	        mat[j][i] = mat[i][j]
	plt.imshow(mat)  
	plt.xlabel("Image Labels")
	plt.ylabel("Image Labels")
	plt.title("Classification Accuracy Between Pairs of Digits")
	plt.show()
	print("Program Finished")

if __name__ == "__main__":
    main()

	


