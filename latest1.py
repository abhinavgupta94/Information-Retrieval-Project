import urllib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.stem
import operator
import wikipedia
import collections
from nltk.util import bigrams
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import load_files
import collections as C
from nltk.collocations import *
trigram_measures = nltk.collocations.TrigramAssocMeasures()
from nltk.util import trigrams
from nltk.util import ngrams
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt




##############
# clustering #
##############

dataset_files = load_files("./20_newsgroups.parent")

#number of clusters 
true_k = 1

f = open("dataset_files_output.txt","wb")
f.write(str(dataset_files))
f.close()


vectorizer = TfidfVectorizer(stop_words='english',decode_error='ignore')
X = vectorizer.fit_transform(dataset_files.data)
model = KMeans(n_clusters=true_k, init='k-means++', n_init=50, max_iter=1000)
model.fit(X)

clusters = C.defaultdict(list)

print "length is ",len(dataset_files.filenames)
k = 0
for i in model.labels_:
    clusters[i].append(dataset_files.filenames[k])  
    k += 1

#print "clusters = ",clusters

list_of_lists = []
for clust in clusters:
    print "\n***********************************\n"
    print "clust = ",clust
    list_of_data_in_cluster = []
    #print "length of this cluster : ",len(clust)
    for i in clusters[clust]:
        print i
        file_open = open(i)
        list_of_data_in_cluster.append(file_open.read())
        file_open.close()
    list_of_lists.append(list_of_data_in_cluster)
        


################################################################################
#opening files in atheism folder and putting each document as an element in docs
################################################################################


docs = ["Lorem Ipsum is simply good dummy","is Lorem good ipsum","good bad simply","good ipsum dummy"]
docs = ["hota bad","anurag anurag knows PJ, the really bad anurag ones. Really really anurag.","Anurag bad","bad"]
docs = ["Lorem Ipsum is simply dummy","Lorem ipsum is good"]
docs = ["Lorem Ipsum is simply dummy good","Lorem ipsum"]   
#docs = ["information systems","computer relevance","restart boot","core 2 duo networks software computer","logic system"]
print "docs array made."

for x in range(0,len(list_of_lists)):

    print "=============== x = ",x," ================"
    docs = list_of_lists[x]
    countDocuments = len(docs)


    #############################################
    # Extending Tfidf to have only stemmed features
    #############################################
    english_stemmer = nltk.stem.SnowballStemmer('english')

    class StemmedTfidfVectorizer(TfidfVectorizer):
        def build_analyzer(self):
            analyzer = super(TfidfVectorizer, self).build_analyzer()
            #print "Analyzer"+analyzer+"\n\n\n"
            # return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))    #FOR STEMMING, UNCOMMENT THIS.
            return lambda doc: (w for w in analyzer(doc))

    tfidf = StemmedTfidfVectorizer(min_df=1, stop_words='english', analyzer='word', decode_error='ignore', ngram_range=(1,1))

    print "stemming done."

    Xv = tfidf.fit_transform(docs) 

    print "Xv matrix done."

    ##################################################################
    # Writing the keywords, whole text, and the tf-idf matrix in file
    ##################################################################

    feature_names = tfidf.get_feature_names()



    f = open("output-dummyIndexing.txt",'w')
    f.write(str(feature_names))
    f.write("\n")
    f.write("\n")
    f.write("\n")

    f.write(str(docs))
    f.write("\n")
    f.write("\n")
    f.write("\n")
    f.write(str(Xv))
    f.write("\n")
    f.write("\n")
    f.write("\n")


    ##########################################
    # FINDING Inverted Indexing as we need it
    ##########################################

    inverseIndexingDict = {}
    #print "num of features",len(feature_names) 

    for j in range(0,len(feature_names)):
        inverseIndexingDict[feature_names[j]] = []
        for i in range(0,countDocuments):
            inverseIndexingDict[feature_names[j]].append(Xv[i,j])

    g = open("inverseIndexing-dummy.txt","w")
    g.write(str(inverseIndexingDict))
    g.close()

    print "inverseIndexingDict done"
    print "len(inverseIndexingDict) = ",len(inverseIndexingDict)

    newInverseIndexingDict = {}
    ########################################################################
    # CHOOSING only those terms which are present in more than len(docs)/5
    # and they should each have threshold > 0.1
    ########################################################################
    for j in range(0,len(feature_names)):
        a = inverseIndexingDict[feature_names[j]]       # a is a list of tf-idf values for a particular keyword with all docs
        count_nonzero = 0
        for item in a:
            if item != 0:
                count_nonzero += 1
        if count_nonzero > 3:
            flag_threshold = 1  #assuming it holds
            # for item in a:                      # item would be one particular tf-idf value
               # if item != 0 and item < 0.1:
                   # flag_threshold = 0
            if flag_threshold == 1:
                newInverseIndexingDict[feature_names[j]] = inverseIndexingDict[feature_names[j]]


    print "newInverseIndexingDict done "
    print "len(newInverseIndexingDict) = ",len(newInverseIndexingDict)
    filename = "newInverseIndexingDict"+str(x)+".txt"
    file1 = open(filename,"wb")
    file1.write(str(newInverseIndexingDict.keys()))
    file1.write("\n\n\n\n")
    file1.write(str(newInverseIndexingDict))
    file1.close() 

    print "Done with matrix and inverted indexing and centroid."

    ##########################################
    # Finding centroid of all
    ##########################################
    centroid = []
    for i in range(0,countDocuments):
        sum = 0
        # for key in newInverseIndexingDict.keys():
            # sum += newInverseIndexingDict[key][i]
        # centroid.insert(i,sum/len(feature_names))
        centroid.insert(i,0)
    
    print str(centroid)

    print "centroid done."

    ############# inverseIndexingDict details ################
    ### keys are terms
    ### values is a list of tfidf with each document
    ##########################################################


    ##############################################
    ############ FINDING TOP K ###################
    ##############################################


    termCentroidDistDict = {}
    for term in newInverseIndexingDict.keys():
        dist = 0
        for i in range(0,countDocuments):
            dist += (centroid[i]-newInverseIndexingDict[term][i])**2
        # dist = ((centroid[0]-newInverseIndexingDict[term][0])**2 + (centroid[1]-newInverseIndexingDict[term][1])**2)**(0.5)
        termCentroidDistDict[term] = dist

    #print "The term to centroid distance is"
    #print termCentroidDistDict      #key : term, value : euclidean distance from centroid (considering term as a vector

    sorted_centroid_dist = sorted(termCentroidDistDict.items(), key=operator.itemgetter(1), reverse=True)

    filename2 = "sorted_centroid_dist"+str(x)+".txt"
    file2 = open(filename2,"wb")
    file2.write(str(sorted_centroid_dist))
    file2.close()

    #print "sorted_centroid_dist= ",sorted_centroid_dist


    #print "Done with term centroid distance AND the top k terms too."
    #print "DONE DONE DONE. Check these files : output-dummyIndexing, inverseIndexing-dummy, and topK-dummy"
    word = []
    value = []
    for i in range(len(sorted_centroid_dist)):
        #word.append((sorted_centroid_dist[i])[0])
        word.append(i)
        value.append((sorted_centroid_dist[i])[1])


    j = len(sorted_centroid_dist)
    median_value = value[j/2] 
    print(median_value)
    print(np.mean(value))

    plt.scatter(word,value)
    plt.ylabel('distance')
    plt.xlabel('word')
    #plt.show()
    plt.savefig("graph.png")




    ############################################
    #Wikipedia
    ############################################

    #Assuming k = 3
    k = 15
    top_k_terms = []
    for i in range(min(k,len(sorted_centroid_dist))):
        top_k_terms.append(str(sorted_centroid_dist[i][0]))
        
    print("Top k terms from feature selection are:")    
    print(top_k_terms)
    print("\n")
        
    eightlist = []
    
    
    for i in range(len(top_k_terms)):
        term = top_k_terms[i]      
        keys = wikipedia.search(term)
        newlist = []
        #keys = wikipedia.search(str(can[0]))
        try:
            '''
            for i in range(len(keys)):
                newlist.append(str(keys[i]))
                page = wikipedia.page(str(keys[i]))
                templist = page.categories
                for j in range(len(templist)):
                    newlist.append(templist[j])
            '''
            newlist.append(str(keys[0]))
            page = wikipedia.page(str(keys[0]))
            templist = page.categories
            for j in range(len(templist)):
                newlist.append(templist[j])
        
        except wikipedia.exceptions.DisambiguationError:
            pass

        try:    
            latestlist = [i for i in newlist if len(str(i).split())<=2]   
        except UnicodeEncodeError:
            pass
        
            
        #print(latestlist)
        #eightlist.append(latestlist)
        for j in range(len(latestlist)):
            eightlist.append(str(latestlist[j]))
        
         
    
    print(eightlist)

    
    finder = TrigramCollocationFinder.from_words(eightlist)
    scored = finder.score_ngrams(trigram_measures.raw_freq)
    print(sorted(trigram for trigram, score in scored))  
    #print [(item, tri_tokens.count(item)) for item in sorted(set(tri_token))]
    
    #finder.apply_freq_filter(3)
    # print(bigram_measures.pmi)
    #print(finder.nbest(trigram_measures.pmi, 1))