from itertools import chain
from collections import defaultdict
import numpy
import pickle
import sys
import copy

with open("tfidf.dict",'rb') as f:
	tfidf = pickle.load(f)
	
with open("tokInfo.dict",'rb') as f:
	tokInfo = pickle.load(f)

with open("pageRank.dict",'rb') as f:
	pageRankDict = pickle.load(f)




print("Normalizing tf idf...",end="")
tfidfNorm = copy.deepcopy(tfidf)
 #TO COMPLETE
 #TO COMPLETE
 #TO COMPLETE
 #TO COMPLETE
		
 #TO COMPLETE
 #TO COMPLETE
 #TO COMPLETE
print("done.")



# Returns the topN documents by token relevance (vector model)
def getBestResults(queryStr, topN, tfidfMatrix):
	query = queryStr.split(" ")
	res = defaultdict(float)
 #TO COMPLETE
 #TO COMPLETE
 #TO COMPLETE
 #TO COMPLETE


# Sorts a list of results according to their pageRank
def rankResults(results):
 #TO COMPLETE
 #TO COMPLETE
 #TO COMPLETE


def printResults(rankedResults):
	for idx,page in enumerate(rankedResults):
		print(str(idx+1) + ". " + page)



query = "darwin" # or sys.argv[1]
top = 15			 # number of results to show

print("Results for ",query,"\n===========")
results = getBestResults(query,top,tfidf)
printResults(results)

print("\n\nResults after normalization for ",query,"\n===========")
results = getBestResults(query,top,tfidfNorm)
printResults(results)


print("\n\nResults after ranking for ",query,"\n===========")
 #TO COMPLETE
 #TO COMPLETE

#bestPageSimilarity = list(reversed([ searchRes[i] for i in numpy.argsort(searchRes)[-10:] ]))
#bestPageSimilarity


