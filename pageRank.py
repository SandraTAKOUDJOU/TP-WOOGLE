from itertools import chain
import numpy
import pickle

CONVERGENCE_LIMIT = 0.0000001

# Load the link information
with open("links.dict",'rb') as f:
	links = pickle.load(f)

# List of page titles
allPages = list(set().union(chain(*links.values()), links.keys()))
# For simplicity of coding we give an index to each of the pages.
linksIdx = [ [allPages.index(target) for target in links.get(source,list())] for source in allPages ]


# Remove redundant links (i.e. same link in the document)
for l in links:
	links[l] = list(set(links[l]))


# One click step in the "random surfer model"
# origin = probability distribution of the presence of the surfer (list of numbers) on each of the page
def surfStep(origin, links):
	dest = [0.0] * len(origin)
	for idx, proba in enumerate(origin):
		if len(links[idx]):
			w = 1.0 /#TO COMPLETE (1 expression)
		else:
			w = 0.0
		for link in links[idx]:
			dest[link] += proba*w
	return dest # proba distribution after a click





# Init of the pageRank algorithm
pageRanks = [1.0/len(allPages)] * len(allPages) # will contain the page ranks
delta = float("inf")
sourceVector = [1.0/len(allPages)] * len(allPages) # default source
# Or use a personalized source vector :
 #TO COMPLETE
 #TO COMPLETE
 #TO COMPLETE
 #TO COMPLETE




while delta >#TO COMPLETE (1 expression)
	print("Convergence delta:",delta,sum(pageRanks),len(pageRanks))
	pageRanksNew =#TO COMPLETE (1 expression)
	jumpProba = sum(pageRanks) - sum(pageRanksNew) # what effect is detected here?
	if jumpProba < 0: # Technical artifact due to numerical errors
		jumpProba = 0
	# Correct for this effect:
	pageRanksNew = [ pageRank + jump for pageRank,jump in zip(pageRanksNew,(p*jumpProba for p in sourceVector)) ]
	# Compute the delta:
 #TO COMPLETE
	pageRanks = pageRanksNew

bestPages = [ allPages[i] for i in numpy.argsort(pageRanks)[-20:] ]
bestPageRanks = [ pageRanks[i] for i in numpy.argsort(pageRanks)[-20:] ]

# Name the entries of the pageRank vector
pageRankDict = dict()
for idx,pageName in enumerate(allPages):
	pageRankDict[pageName] = pageRanks[idx]

# Rank of some pages:
 #TO COMPLETE
 #TO COMPLETE
 #TO COMPLETE



# Save the ranks as pickle object
with open("pageRank.dict",'wb') as fileout:
	pickle.dump(pageRankDict, fileout, protocol=pickle.HIGHEST_PROTOCOL)

