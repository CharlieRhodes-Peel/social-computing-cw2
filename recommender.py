# ---- Imports ----
import csv
import math
import time
import random
import numpy as np
# -----------------


def trainingStart(fileName):
    """
    Starts the training by getting all the users and items into a set, also calcs
    global mean while we're here

    Returns:
        userIDs (set), itemIDs (set), globalMean (float), count (int)
    """

    print("first pass...")

    userIDs = set()
    itemIDs = set()
    runningRatingTotal = 0

    count = 0

    with open (fileName, "r") as file:
        reader = csv.reader(file)
        for row in reader:

            # Just to make sure!
            if not row:
                continue

            # Put into respective sets
            userIDs.add(int(row[0]))
            itemIDs.add(int(row[1]))
            runningRatingTotal += float(row[2])

            count += 1

    globalMean = runningRatingTotal / count

    print(f"Global Mean calculated: {globalMean}")
    return userIDs, itemIDs, globalMean, count

def setupLatentFactorVectors(nUsers, nItems, nFactors):

    #Setting up matricies in latent factor space
    userLatentMatrix = np.random.normal(0.0, 0.1, (nUsers, nFactors))
    itemLatentMatrix = np.random.normal(0.0, 0.1 (nItems, nFactors))

    #Setting up biases
    userBiasMatrix = np.zeros(nUsers, dtype=np.float32)
    itemBiasMatrix = np.zeros(nItems, dtype=np.float32)

    return userLatentMatrix, itemLatentMatrix, userBiasMatrix, itemBiasMatrix

def sgd(fileName, nIterations, globalMean, uLatent, iLatent, uBias, iBias, learningRate, regularisation):

    for i in range(nIterations):
        iStartTime = time.time()
        with open (fileName, "r") as f:
            reader = csv.reader(f)
            
            errorSum = 0
            count = 0

            for row in reader:

                # Just to make sure!
                if not row:
                    continue
                
                uID = int(row[0])
                iID = int(row[1])
                rating = float(row[2])

                # Predict and then calc error
                pred = globalMean + uBias[uID] + iBias[i] + np.dot(uLatent[uID], iLatent[iID])
                error = rating - pred

                #Update the factors
                prevUserFactors = uLatent[uID]

                uLatent[uID] += learningRate * (error * iLatent[iID] - regularisation * uLatent[uID])
                iLatent[iID] += learningRate * (error * prevUserFactors - regularisation * iLatent[iID])


                #Update the biases
                uBias[uID] += learningRate * (error - regularisation * uBias[uID])
                iBias[iID] += learningRate * (error - regularisation * iBias[iID])

                #Counts
                errorSum += error
                count += 1
            
        print(f"iteration {i} complete | current error: {errorSum / count} | time took: {time.time() - iStartTime}s")
    return None


###### ACTUAL RUNNING STUFF NO MORE FUNCTIONS #########
TRAINING_FILE = "train_20M_withratings.csv"
LATENT_FACTORS = 50
N_ITERATIONS = 20

# Do a first pass an calculated global mean and sort stuff into sets
userIDs, itemIDs, globalMean, rowCount = trainingStart(TRAINING_FILE)

#Setup the latent factors based on the sizes found previous
uLatent, iLatent, uBias, iBias = setupLatentFactorVectors(len(userIDs), len(itemIDs), LATENT_FACTORS)

sgd = (TRAINING_FILE, N_ITERATIONS, userIDs, itemIDs, globalMean, uLatent, iLatent, uBias, iBias)


