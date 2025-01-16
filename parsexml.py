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

xmlFiles = list(chain(*[ glob.glob(globName)  for globName in sys.argv[1:] ]))
print("Files as input:", xmlFiles)

docs = dict()

##############################
print("Parsing XML...")
##############################
for xmlFile in xmlFiles:
	pages = xml.etree.ElementTree.parse(xmlFile).getroot()

	for page in pages.findall("{http://www.mediawiki.org/xml/export-0.10/}page"):
		titles = page.findall("{http://www.mediawiki.org/xml/export-0.10/}title")
		revisions = page.findall("{http://www.mediawiki.org/xml/export-0.10/}revision")
	
		if titles and revisions:
			revision = revisions[0] # last revision
			contents = revision.findall("{http://www.mediawiki.org/xml/export-0.10/}text")
			if contents:
				docs[titles[0].text] = contents[0].text 



# Some regEx for parsing
cleanExtLinks = "\[[.]+\]|{{[.]+}}"   #TO COMPLETE (1 expression)
linkRe = "\[\[([^\|]+)(\|[^\|])+\]\]"  #TO COMPLETE (1 expression)
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


tokInfo = defaultdict(float) # tokInfo[tok] contains the information in bits of the token
tf = dict() # tf[doc][tok] contains the frequency of the token tok in document doc
  

for doc, toks in doctok.items(): # Building tf table
    comptes_tok = Counter(toks)
    
    tf_doc = {tok : comptes_tok[tok] / len(toks) for tok in comptes_tok}
    tf[doc] = tf_doc
    

idf = dict() # Building idf table    

wordset = set()
N = len(doctok)

for toks in doctok.values():
    wordset.update(toks)
    

for tok in wordset:
    N_tok = sum([1 for toks in doctok.values() if tok in toks])  
    if N_tok !=0:
       idf[tok]= - math.log(N_tok/N)
       
    

print("done.")

print("creating tf-idf...",end="")
tfidf = defaultdict(dict) # this should be in reverse sparse format

#*****************************************************PARSE INDEX ***********************************************************************************

for doc, tf_doc in tf.items():
    tfidf[doc] = {tok : tf_doc[tok] * idf[tok] for tok in tf_doc}


#*****************************************************REVERSE PARSE INDEX ***********************************************************************************

re_tfidf = dict()
for doc, toks_score in tfidf.items():
    for tok, score in toks_score.items():
        if tok not in re_tfidf:
            re_tfidf[tok] = dict()
        re_tfidf[tok][doc] = score
            
            
 
print("done.")


print("Saving the links and the tfidf as pickle objects...")
with open("links.dict",'wb') as fileout:
	pickle.dump(links, fileout, protocol=pickle.HIGHEST_PROTOCOL)

with open("tfidf.dict",'wb') as fileout:
	pickle.dump(tfidf, fileout, protocol=pickle.HIGHEST_PROTOCOL)

with open("tokInfo.dict",'wb') as fileout:
	pickle.dump(tokInfo, fileout, protocol=pickle.HIGHEST_PROTOCOL)

end = time.time()


print('DURATION = ', end - start)