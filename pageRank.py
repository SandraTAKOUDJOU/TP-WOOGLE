from itertools import chain
import numpy
import pickle

CONVERGENCE_LIMIT = 0.0000001
links = {}
# Load the link information
try:
    with open("links.dict", 'rb') as filein:
        links = pickle.load(filein)
        print("Fichier links.dict chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement de 'links.dict': {e}")


# List of page titles
allPages = list(set().union(chain(*links.values()), links.keys()))
# For simplicity of coding we give an index to each of the pages.
linksIdx = [ [allPages.index(target) for target in links.get(source,list())] for source in allPages ]


# Remove redundant links
for l in links:
	links[l] = list(set(links[l]))
	

# One click step in the "random surfer model"
# origin = probability distribution of the presence of the surfer (list of numbers) on each of the page
alpha = 0.85  # Probability of following a link
def surfStep(origin, links):
	dest = [0.0] * len(origin)
	for idx, proba in enumerate(origin):
		if len(links[idx]):
			w = alpha / len(links[idx]) if len(links[idx]) > 0 else 0.0
		for link in links[idx]:
			dest[link] += proba * w
	return dest # proba distribution after a click






## Init of the pageRank algorithm
pageRanks = [1.0/len(allPages)] * len(allPages) # will contain the page ranks
delta = float("inf")
sourceVector = [1.0/len(allPages)] * len(allPages) # default source
sourceVector[allPages.index("Page A")] = 0.9

iterations = 0
while delta > CONVERGENCE_LIMIT:
    print("Convergence delta:", delta, sum(pageRanks), len(pageRanks))
    pageRanksNew = surfStep(pageRanks, links)
    jumpProba = sum(pageRanks) - sum(pageRanksNew)
    if jumpProba < 0:
        jumpProba = 0
    pageRanksNew = [pageRank + jump for pageRank, jump in zip(pageRanksNew, (p * jumpProba for p in sourceVector))]
    delta = max([abs(pageRanks[i] - pageRanksNew[i]) for i in range(len(pageRanks))])
    pageRanks = pageRanksNew
    iterations += 1

print(f"Total iterations: {iterations}")

# Create pageRankDict
pageRankDict = {allPages[i]: pageRanks[i] for i in range(len(allPages))}

# Rank of DNA
print(f"Rank of 'DNA': {pageRankDict.get('DNA')}")

# Page with the highest rank
max_rank_idx = numpy.argmax(pageRanks)
highest_rank_page = allPages[max_rank_idx]
print(f"The page with the highest rank is: {highest_rank_page}")

# Save the ranks as pickle object
with open("pageRank.dict",'wb') as fileout:
	pickle.dump(pageRankDict, fileout, protocol=pickle.HIGHEST_PROTOCOL)
