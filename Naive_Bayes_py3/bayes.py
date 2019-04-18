# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from numpy import *
import math
import numpy as np
from functools import reduce
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
               ['maybe','not','take','him','to','dog','park','stupid'],
               ['my','dalmation','is','so','cute','I','love','him'],
               ['stop','posting','stupid','worthless','garbage'],
               ['mr','licks','ate','my','steak','how','to','stop','him'],
               ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:print("the world:%s is not in my Vocabulary!"% word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)							#计算训练的文档数目
	numWords = len(trainMatrix[0])							#计算每篇文档的词条数
	pAbusive = sum(trainCategory)/float(numTrainDocs)		#文档属于侮辱类的概率
	p0Num = np.zeros(numWords); p1Num = np.zeros(numWords)	#创建numpy.zeros数组,
	p0Denom = 0.0; p1Denom = 0.0                        	#分母初始化为0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:							#统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:												#统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	p1Vect = p1Num/p1Denom									#相除        
	p0Vect = p0Num/p0Denom          
	return p0Vect,p1Vect,pAbusive							#返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V ,pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry , 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry=['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry , 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def textParse(bigString):
    import re
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList=[];classList=[];fullText=[]
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=list(range(50));testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
        trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(array(trainMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is:',float(errorCount)/len(testSet))


def calcMostFreq(vocabList,fullText):  
    import operator  
    freqDict = {}  
    for token in vocabList:  
        freqDict[token]=fullText.count(token)  
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)   
    return sortedFreq[:50]         
  
def localWords(feed1,feed0):  
    import feedparser  
    docList=[]; classList = []; fullText =[]  
    minLen = min(len(feed1['entries']),len(feed0['entries']))  
    for i in range(minLen):  
        wordList = textParse(feed1['entries'][i]['summary'])  
        docList.append(wordList)  
        fullText.extend(wordList)  
        classList.append(1) #NY is class 1  
        wordList = textParse(feed0['entries'][i]['summary'])  
        docList.append(wordList)  
        fullText.extend(wordList)  
        classList.append(0)  
    vocabList = createVocabList(docList)#create vocabulary  
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words  
    for pairW in top30Words:  
        if pairW[0] in vocabList: vocabList.remove(pairW[0])  
    trainingSet = list(range(2*minLen)); testSet=[]           #create test set  
    for i in range(20):  
        randIndex = int(random.uniform(0,len(trainingSet)))  
        testSet.append(trainingSet[randIndex])  
        del(trainingSet[randIndex])    
    trainMat=[]; trainClasses = []  
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0  
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))  
        trainClasses.append(classList[docIndex])  
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))  
    errorCount = 0  
    for docIndex in testSet:        #classify the remaining items  
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])  
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:  
            errorCount += 1  
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V,top30Words 

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V,top30Words=localWords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-6.0:topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-6.0:topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair: pair[1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY=sorted(topNY,key=lambda pair: pair[1],reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
p0v,p1v,pAb = trainNB0(trainMat, listClasses)
print("任意文档属于侮辱性文档的概率:",pAb)
print("10封随机选择的电子邮件的分类错误率：")
spamTest()

        