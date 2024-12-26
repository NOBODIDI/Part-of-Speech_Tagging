import os
import sys
import argparse
from copy import deepcopy
import numpy as np
import time

e = 0.00101

TAGS = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
        "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
        "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
        "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
        'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
        'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
        'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
        'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
        'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']


def read_training_files(training_list):
    """
    Reads the training files and returns a list of words with their tags.
    """
    fileToWord = time.time()

    words = []
    counter = 0
    for file in training_list:
        f = open(file)
        lines = f.readlines()
        for l in lines:
            l = str.strip(str(l))
            words.append(l.split(" : "))
            counter += 1
        f.close()
    
    return words

def getM_fTag(words):
    """
    Creates the M and fTag dictionaries from the list of words.
    """
    wordToDict = time.time()

    M = dict()
    fTag = [0] * len(TAGS)
    knownWds = dict()

    for i in range(len(TAGS)): 
        M[TAGS[i]] = dict()
        M[TAGS[i]]["TOT"] = 0
    
    for i in range(len(words)):
        knownWds[words[i][0]] = 0
        POSWord = TAGS.index(words[i][1])
        fTag[POSWord] += 1
        if words[i][0] not in M[TAGS[POSWord]]:
            for POS in TAGS:
                M[POS][words[i][0]] = e
        M[TAGS[POSWord]][words[i][0]] += 1
        M[TAGS[POSWord]]["TOT"] += 1
            

    for POS in TAGS:
        for word in M[POS]:
            if M[POS]["TOT"] != 0:
                M[POS][word] = M[POS][word] / M[POS]["TOT"]
    
    return M, fTag, knownWds

def getI(words):
    """
    Creates the initial probability matrix I.
    """
    wordToInit = time.time()

    wCount = 0
    initArr = [e]*len(TAGS)
    for i in range(len(words)):
        if  (i == 0 or (words[i][0] in [".", "?", "!", "-"]) and i != len(words) - 1):
            initArr[TAGS.index(words[i + 1][1])] += 1
            wCount += 1
    for i in range(len(TAGS)):
        initArr[i] = initArr[i] / wCount
    
    I = np.zeros((1, len(TAGS)))
    for i in range(len(TAGS)): 
        I[0, i] = max(initArr[i], e)
    
    return I

def getT(words):
    """
    Creates the transition probability matrix T.
    """
    wordToTrans = time.time()

    tran = [[e for i in range(len(TAGS) + 1)] for j in range(len(TAGS))]
    for k in range(len(words) - 1):
        word = words[k][1]
        next_word = words[k + 1][1]
        i = TAGS.index(word)
        j = TAGS.index(next_word)
        tran[i][j] += 1
        tran[i][len(TAGS)] += 1 + e * len(TAGS)
    
    T = np.zeros((len(TAGS), len(TAGS)))
    for i in range(len(tran)):
        for j in range(len(tran) - 1):
            tran[i][j] = tran[i][j] / tran[i][len(TAGS)]
            T[i][j] = tran[i][j]

    return T

def getDistTag(fTag, nbWords):
    """
    Calculates the distribution of tags.
    """
    for i in range(len(fTag)):
        fTag[i] = fTag[i] / nbWords
    
    return fTag


def read_testing_file(file):
    """
    Reads the testing file and returns a list of sentences.
    """
    with open(file, 'r') as f:
        testWds = f.read().splitlines()
    E = []
    temp = []
    for j in range (len(testWds)):
        temp.append(testWds[j])
        if testWds[j] in ['.', '?', '!', '-']:
            E.append(deepcopy(temp))
            temp = []            
    
    return E

def doViterbi(distTag, sent, I, T, M, knownWds): 
    """
    Performs the Viterbi algorithm to find the most probable sequence of tags for a sentence.
    """
    tagsForSent = []
    prob = np.zeros((len(sent), len(TAGS)))
    prev = np.zeros((len(sent), len(TAGS)))

    for i in range(len(TAGS)):
        if sent[0] in M[TAGS[i]]:
            prob[0,i] = I[0,i] * M[TAGS[i]][sent[0]]
        else:
            prob[0, i] = I[0, i] * (1 / len(TAGS)) 
        prev[0, i] = None

    for t in range(1, len(sent)):
        if sent[t] in knownWds:
            for i in range(len(TAGS)):
                if sent[t] in M[TAGS[i]]:
                    m = M[TAGS[i]][sent[t]]
                else:
                    m = 1 / (len(TAGS) - 1)
                temp = prob[t - 1, :] * T[:, i] * m
                maxP = np.max(temp)
                x = np.argmax(temp)
                prob[t, i] = maxP
                prev[t, i] = x
        else:
            for i in range(len(TAGS)):
                m = distTag[i]
                temp = prob[t - 1, :] * T[:, i] * m
                maxP = np.max(temp)
                x = np.argmax(temp)
                prob[t, i] = maxP
                prev[t, i] = x
        prob[t, :] = prob[t, :] / np.sum(prob[t, :])
    xP = np.argmax(prob[len(sent) - 1, :])
    tagsForSent.append(TAGS[int(xP)])
    for i in range(len(sent) - 1, 0, -1):
        xP = prev[i, int(xP)]
        tagsForSent.append(TAGS[int(xP)])
    tagsForSent.reverse()
    
    return tagsForSent


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    print("Starting the tagging process.")
    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))
    print("test file is {}".format(args.testfile))
    print("output file is {}".format(args.outputfile))
    
    startTime = time.time()

    words = read_training_files(training_list)
    M, fTag, knownWds = getM_fTag(words)
    I = getI(words)
    T = getT(words)
    distTag = getDistTag(fTag, len(words))

    startTag = time.time()

    E = read_testing_file(args.testfile)
    S = []

    for sent in E: 
        S.append(doViterbi(distTag, [wd for wd in sent], I, T, M, knownWds))

    outFile = open(args.outputfile, "w")
    for i in range(len(E)):
        for j in range(len(E[i])):
                    outFile.write("{} : {}\n".format(E[i][j], S[i][j]))
    outFile.close()

