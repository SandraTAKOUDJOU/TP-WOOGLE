import sys
import xml.etree.ElementTree
from collections import defaultdict
import numpy
import re
import pickle
import glob
from itertools import chain
import time
from collections import Counter
import math



start = time.time()

#xmlFiles = list(chain(*[ glob.glob(globName)  for globName in sys.argv[1:] ]))
xmlFiles = ["dws\\xaa.chunks"]
print("Files as input:", xmlFiles)

docs = dict()

##############################
print("Parsing XML...")
##############################
for xmlFile in xmlFiles:
	pages = xml.etree.ElementTree.parse(xmlFile).getroot()

	for page in pages.findall("{http://www.mediawiki.org/xml/export-0.11/}page"):
		titles = page.findall("{http://www.mediawiki.org/xml/export-0.11/}title")
		revisions = page.findall("{http://www.mediawiki.org/xml/export-0.11/}revision")
	
		if titles and revisions:
			revision = revisions[0] # last revision
			contents = revision.findall("{http://www.mediawiki.org/xml/export-0.11/}text")
			if contents:
				docs[titles[0].text] = contents[0].text 

print(pages)

# Some regEx for parsing
cleanExtLinks = "\[[.]+\]|{{[.]+}}"   #TO COMPLETE (1 expression)
linkRe = "\[\[([^\|])+(\|[^\|])*\]\]"  #TO COMPLETE (1 expression)
removeLinkRe = "\[\[[^\]]+\|([^\|\]]+)\]\]"
removeLink2Re =  "\[\[([^\|\]]+)\]\]"
wordRe = "[a-zA-Z\-]+"
stopWords = ["-"]


print("Extracting links, transforming links in text, tokenizing, and filling a tok-doc matrix...")
links = dict()
doctok = dict()
for idx,doc in enumerate(docs):
	if idx%(len(docs)//20) == 0:
		print("Progress " + str(int(idx*100/len(docs)))  +"%")
	links[doc] = list()	
	for link in re.finditer(linkRe,docs[doc]):
		target = link.group(1).split('|')[0]
		if target in docs.keys():
			#print(doc + " --> " + target)
			links[doc] += [target]
			links[doc] += [target]
			
	cleanDoc = re.sub(cleanExtLinks,"",docs[doc])

	# transform links to text
	docs[doc] = re.sub(removeLinkRe,r"\1",cleanDoc)
	docs[doc] = re.sub(removeLink2Re,r"\1",docs[doc])
	
	# fill the doctok matrix
	doctok[doc] = list()
	for wordre in re.finditer(wordRe,cleanDoc):
		word = wordre.group(0).lower()
		if word not in stopWords:
			doctok[doc] += [word]



print("done.")

print("Building tf-idf table...")
docList = doctok.keys()
Ndocs = len(docList)

#************************************************************INFORMATION IN BITS OF THE TOKEN*******************************************************#
term_freq = defaultdict(int)
total_terms = 0
tokInfo = defaultdict(float) # tokInfo[tok] contains the information in bits of the token

for doc, toks in doctok.items():
    for tok in toks:
        term_freq[tok] += 1
        total_terms += 1

for tok, count in term_freq.items():
    prob = count / total_terms
    tokInfo[tok] = -math.log2(prob)

#for tok, info in tokInfo.items():
#    print(f"Token: {tok}, Information (bits): {info:.4f}")

#*****************************************************WITHOUT REVERSE PARSE INDEX ***********************************************************************************
"""
# Building tf table
tf = dict() # tf[doc][tok] contains the frequency of the token tok in document doc
  
for doc, toks in doctok.items(): 
    comptes_tok = Counter(toks)
    tf_doc = {tok : comptes_tok[tok] / len(toks) for tok in comptes_tok}
    tf[doc] = tf_doc
    
# Building idf table
idf = dict()     
wordset = set()
N = len(doctok)

for toks in doctok.values():
    wordset.update(toks)
for tok in wordset:
    N_tok = sum([1 for toks in doctok.values() if tok in toks])  
    if N_tok > 0:
       idf[tok]= - math.log(N_tok/N)
       
print("done.")
print("creating tf-idf...",end="")
tfidf = defaultdict(dict) 

for doc, tf_doc in tf.items():
    tfidf[doc] = {tok : tf_doc[tok] * idf[tok] for tok in tf_doc}

"""
#*****************************************************WITH REVERSE PARSE INDEX ***********************************************************************************

wordset = set([tok for toks in doctok.values() for tok in toks])

N = len(doctok)

inverse_index = defaultdict(set)
for doc, toks in doctok.items():
    unique_tokens = set(toks)  
    for tok in unique_tokens:
        inverse_index[tok].add(doc)

# Building tf table
tf = defaultdict(dict)
for doc, toks in doctok.items():
    doc_len = len(toks)
    for tok in set(toks):  
        tf[doc][tok] = toks.count(tok) / doc_len  

# Building idf table
idf = {}
for tok in wordset:
    N_tok = len(inverse_index.get(tok, []))  
    if N_tok > 0:
        idf[tok] = - math.log(N_tok/N)

# Calcul de la matrice TF-IDF
tfidf = defaultdict(dict)
for doc in doctok:
    for tok in tf[doc]:
        tfidf[doc][tok] = tf[doc][tok] * idf.get(tok, 0)  
        
"""
#re_tfidf = dict()
#for doc, toks_score in tfidf.items():
#    for tok, score in toks_score.items():
#        if tok not in re_tfidf:
#            re_tfidf[tok] = dict()
"""  

print("done.")


print("Saving the links and the tfidf as pickle objects...")
with open("links.dict",'wb') as fileout:
	pickle.dump(links, fileout, protocol=pickle.HIGHEST_PROTOCOL)

with open("tfidf.dict",'wb') as fileout:
	pickle.dump(tfidf, fileout, protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(tfidf, fileout, protocol=pickle.HIGHEST_PROTOCOL)
 
with open("tokInfo.dict",'wb') as fileout:
	pickle.dump(tokInfo, fileout, protocol=pickle.HIGHEST_PROTOCOL)

end = time.time()

print('DURATION = ', end - start)