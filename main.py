import os
import math
import re
import pandas as pd
from collections import defaultdict, Counter
from bs4 import BeautifulSoup as soup
import lxml

#main search engine class that will return relevant docs scraped from the concordia.ca domain
class SearchEngine(object):
    
    def __init__(self, corpus=""):
        
        #hoist corpus, docid,idf,bm25,etc... 
        self.corpus = corpus
        self.docid = dict()
        self.idf = defaultdict(list)
        self.idf_bm25 = defaultdict(list)
        self.tf = defaultdict(list)
        self.tf_bm25 = defaultdict(list)
        self.avgdl = None
        self.tfidf = None
        self.bm25 = None
        self.query_tf_idf = None
        
        #build corpus
        run = self.build_corpus()

    """Tokenize string:
    Takes in a strings and does the following:
    - splits on spaces
    - replace special characters or punctuation at end of words with white sapce
    - normalizes string to lowercase
    - Strips the extra whitespace from begining or end
    - Removes stopword
    - finally Returns a list of processed tokens.
    """
    def tokenize(self, string):
        #Load NLTK stopword list (reads from english_stopwords file in this directory)
        with open('english_stopwords', 'r') as f:
            stopwords = [x.replace('\n', '') for x in f.readlines()]
            
        #Split on spaces
        words = string.split(" ")
        #Regex matches punctuations at the end of words
        pattern1 = r'\W$'
        #instantiate a payload list
        payload = list()
        # Check for sentence to tokenize by list length
        if len(words)>1:
            for word in words:
                #remove space and normalize
                word = word.strip().lower()
                #replace pattern
                word = re.sub(pattern1, '', word)
                #append any word not in stopwords list to payload list
                if word not in stopwords:
                    payload.append(word)
        else:
            #remove space and normalize
            word = words[0].strip().lower()
            word = re.sub(pattern1, '', word)
            if word not in stopwords:
                payload.append(word)

        yield payload
        
        
    """ 
    Takes in a path to an html document.
    - Filters those without section tag with the id content-main
    - Filters all content with script tag
    - Tokenizes sentences by tag
    - Remove cases with special characters or punctuation at the begining
        (These are errors from the parser, new lines, symbols,etc)
    finally returns None or a list of relevant document tokens.
    """
    def html_parser(self, html):
            #Load html document
            with open(html, encoding='UTF-8') as page:  
                # load into parser
                data = soup(page, 'html.parser', from_encoding='UTF-8')
                # find content only that has attribute "conetent-main"
                contentmain = data.find('section', {'id':'content-main'})
                # filter out script tags and documents without content-main
                if contentmain:
                    for s in contentmain.select('script'):
                        s.extract()   
                    # search for html text groups
                    text = contentmain.find_all(text=True)  
                    # Pretokenization filtering takes place on incoming text strings
                    # filter all sentences with special character at the beginign
                    pattern1 = r'^\W'
                    # ending punctuation 
                    pattern2 = r'\W$'
                    payload = list()
                    for sentence in text:
                        if not re.search(pattern1, sentence):
                            # remove white spaces
                            sentence = sentence.strip()
                            # replace punctuation with nothing
                            sentence = re.sub(pattern2, '', sentence)
                            # rebuild list by unpacking tokenization generator
                            words = list(*self.tokenize(sentence))
                            for word in words:
                                    payload.append(word) 
                        else:
                            continue
                    return payload
                else:
                    return
                

    """ 
    Takes in a string query 
        and returns a list that is cleaned and tokenized.
    """
    def query_parser(self, query):
        return list(*self.tokenize(query))
    
                
    """ 
    Takes a list of strings and returns term frequecy (tf) a
    tf = count(term)/len(doc)
    where count(term) is the number of apperances of a word in a document
    len(doc) is the length of a document
    Returns a dictionary of tokens for keys and tf for values.
    """
    def calculate_tf(self, bow, avgdl):
        #instantiate payload tf and bm25 dictionary 
        payload = dict()
        payload_bm25 = dict()
        #length of document
        N = len(bow)
        # Turn bag of words to a set to avoid repeated keys
        for word in set(bow):
            if word:
                f = bow.count(word)
                payload[word] = f / N
                #use bm25 formula
                payload_bm25[word] = (f * (1.5 + 1)) / f + 1.5 * (1 - 0.75 + 0.75 * N / avgdl)
        return payload, payload_bm25 


    """ 
    Takes an integer and list of term counts across all docs. 
    Inverse document frequency (idf) as idf = log(N/n)
    where N = Total number of documents in the corpus
    and n = Number of documents the term occurs in
    and returns a dictionary of words for keys and idf for values.
    """
    def calculate_idf(self, total_docs, term_occurences):
        tfidf = math.log10(total_docs/len(term_occurences))
        bm25 = math.log(total_docs/(len(term_occurences)+0.5)/(len(term_occurences) + 0.5) + 1 )
        return tfidf, bm25
    
    
    """ 
    Takes the term frequency and inverse document frequency dictionaries.
    doc_tfidif = tf * idf
    returns a tf-idf list with docid as index.
    """
    def calculate_doc_tfidf(self, termfreq, invdocfr):
        tf_idf = defaultdict(list)
        # Unpack defaultdict
        for docid, doc in termfreq.items():
            doc = doc[0]
            for term, tf in doc.items():
                # Calculate tfidf
                doc[term] = tf * invdocfr[term]
            tf_idf[docid].append(doc)

        return tf_idf


    """ 
    Takes the query term frequency and inverse document frequency dictionaries.
    query_tfidif = (0.5 + 0.5 * (qt/max(qt) ) ) * idf
    Returns dictionary with query terms as keys and weighted tfidf as values.
    """
    def calculate_query_tfidf(self, parsed_query, invdocfr):
        # Create defaultdict term count
        w_tfidf = {}
        raw_count = {}
        for word in parsed_query:
            raw_count[word] = parsed_query.count(word)
        
        # find max term count value
        max_count = max(raw_count.values())
        # weighted  term frequency inverse document frequency
        for key, value in raw_count.items():
            # check if terms is in documents
            if invdocfr[key]:
                w_tfidf[key] = (0.5 + 0.5*(value/max_count)) * invdocfr[key]
     
        return w_tfidf


    """ Takes the query term frequency and inverse document frequency dictionaries.
    query_tfidif = (0.5 + 0.5 * (qt/max(qt) ) ) * idf
    Returns dictionary with query terms as keys and weighted tfidf as values.
    """
    def calculate_query_bm25(self, parsed_query, invdocfr):
        # Create defaultdict term count
        w_tfidf = {}
        raw_count = {}
        for word in parsed_query:
            raw_count[word] = parsed_query.count(word)
        
        # find max term count value
        max_count = max(raw_count.values())
        # weighted  term frequency inverse document frequency
        for key, value in raw_count.items():
            # check if terms is in documents
            if invdocfr[key]:
                w_tfidf[key] = (0.5 + 0.5*(value/max_count)) * invdocfr[key]
     
        return w_tfidf
            
        
    def build_corpus(self):
        """Builds a dataframe of the corpuses tfidf."""
        # path to spidy files
        path = self.corpus
        # generate path to saved pages
        # ensures to load html
        files = ["".join([path, file]) 
                 for file in os.listdir(path) 
                 if file.split('.')[-1] == 'html']
        
        # create temp idf
        temp_idf = defaultdict(list)
        doc_length = []
        # Parse and compute to build up idf and tf
        for idx, doc_path in enumerate(files):
            # parse document for important text
            bow = self.html_parser(doc_path)
            # skip failed parsing
            if not bow:
                print(f"Content-main is missing from the following document.\n Parsing has failed.\n {doc_path}")
                continue
            else:
                doc_length.append(len(bow))
                
        # To fix the loop issue -- replace with generator
        self.avgdl = sum(doc_length)/ len(doc_length)
        for idx, doc_path in enumerate(files):
            # parse document for important text
            bow = self.html_parser(doc_path)
            # skip failed parsing
            if not bow:
                continue            
            
            #calculate term frequency
            termfrequency, tf_bm25 = self.calculate_tf(bow, self.avgdl)
            # hoist tf to class level
            self.tf[idx].append(termfrequency)
            self.tf_bm25[idx].append(tf_bm25)
            
            # append doc index to list
            self.docid[idx] = doc_path 
            
            # create dictionary key of unique word
            for word in set(termfrequency.keys()):
                temp_idf[word].append(idx)

        # Mutate idf to dictionary of term frequency in all docs
        # create a list to ensure copy

        for key, value in temp_idf.items():
            # Total documents
            total_docs = len(self.docid.keys())
            self.idf[key], self.idf_bm25[key] = self.calculate_idf(total_docs ,value)

        self.tfidf = self.calculate_doc_tfidf(self.tf, self.idf)
        self.bm25 = self.calculate_doc_tfidf(self.tf_bm25, self.idf_bm25)      
        
        print(f"{50*'*'} \n Corpus built.")
        
        return #nothing to return bc i am just building the corpus
 

    """ Takes a string query.
    Returns a tfidf rank sorted Pandas DataFrame with tfidf weights per term.
    The indexs are the html documents with columns labeled as word tokens.
    """
    def query_tfidf(self, string):
        # Parse query string then normalize and filter it
        query = self.query_parser(string)
        query_tfidf = self.calculate_query_tfidf(query, self.idf)
        
        return query_tfidf
    

    """ Takes a string query.
    Returns a tfidf rank sorted Pandas DataFrame with tfidf weights per term.
    The indexs are the html documents with columns labeled as word tokens.
    """
    def query_bm25(self, string):
        # Parse, normalize, and filter query string
        query = self.query_parser(string)
        query_tfidf = self.calculate_query_bm25(query, self.idf_bm25)
        
        return query_tfidf
    
     
    """Accepts a query and ranking method to sort pages.
    Returns a pandas series
    """
    def query(self, q_string, method=""):
        payload = dict()
        if not method:
            method = "tfidf"
         
        if method == "tfidf":
            
            qtfidf = self.query_tfidf(q_string)
            
            tfidf = pd.DataFrame([v[0] for k, v in self.tfidf.items()], index=self.tfidf.keys())

            # Find the most important query term
            keyword = min(qtfidf)
            #aborted design ideas that was returning a dictionary with ALL relevant documents per keyword
            #Sort keywords by weight and create a dict
            #for key, value in sorted(qtfidf.items(), key=lambda x: x[1]):
                #payload[key] = tfidf[key].sort_values()
            # Find min value tfidf from query as keyword and sort the results of that column
            payload = tfidf[keyword].sort_values()
            return payload
        
        if method == "bm25":
            qtfidf = self.query_bm25(q_string)
            tfidf = pd.DataFrame([v[0] for k, v in self.bm25.items()], index=self.bm25.keys())
            self.query_tf_idf = qtfidf
            # Find the most important query term
            keyword = min(qtfidf)
            #Sort keywords by weight and create a dict
            #for key, value in sorted(qtfidf.items(), key=lambda x: x[1]):
            #    payload[key] = tfidf[key].sort_values()
            # Find min value tfidf from query as keyword and sort the results of that column
            payload = tfidf[keyword].sort_values()
            return payload

#end of searchEngine class

"""
To Run the program:
1. I created a virtual environment - run venv
2. In IDE, open a new terminal - run command 'python' to start a python CLI
3. Run command 'import main'
4. Instantiate the spidy path: path = './spidy/spidy/saved/'
5. Insantiate a search query variable: search = main.SearchEngine(path)
6. SearchEngine will start running and the corpus will start being built, wait for 'Corpus built' message
7. Now you can query what you wish. For example: search.query("which researchers at Concordia worked on COVID 19-related research?", 'tfidf') 
this will calculate the bm25 index for the given query string. You can even ommit the second parameter if you want and it will calculate by default
the tf-idf of your query string.
Other possible querying:
- search.docid[5]
- search.idf
- search.idf_bm25
- search.tf_bm25
- search.avgdl
- search.tfidf
- search.bm25

"""


"""SCRAP IDEAS AND TESTS"""

"""
results = defaultdict(list)
count = 0
for key, value in covid.items():
    count +=1
    weight = count / len(covid)
    value = value.dropna()
    for docid, tfidf in value.items():
        if tfidf:
            results[docid].append(tfidf * weight)

sorted(search.query_tf_idf.items(), key=lambda x: x[1])

# create a ranking scheme
res_sum = [(key, sum(value)**2/len(value)) for key, value in results.items()]
# sorted(results.items(), key=lambda x: sum(x[1]), reverse=True)
sorted(res_sum, key=lambda x: x[1])


#Takes the term frequency and inverse document frequency dictionaries.
#doc_tfidif = tf * idf
#Returns a Pandas dataframe with docid as index.

def calculate_doc_tfidf(termfreq, invdocfr):
    tf_idf = defaultdict(list)
    for docid, doc in termfreq.items():
        doc = doc[0]
    
        for term, tf in doc.items():
            print(term, tf)
            doc[term] = tf * invdocfr[term]
            print(term, doc[term])
            
        tf_idf[docid].append(doc)
    
    return tf_idf


"""


