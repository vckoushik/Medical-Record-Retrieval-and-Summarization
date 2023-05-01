from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
import math
from collections import Counter
import operator
import numpy as np
import re
import optparse
import os 
import glob
import sys


class IRModel:
    invertedIndex ={}
    def __init__(self, path2docs):
        self.docno, self.raw_documents = self.extract_text(path2docs)
        self.documents = self.preprocess(self.raw_documents)
        self.vocab = self.get_vocab(self.documents)
        self.N = len(self.documents)        # total number of documents
        self.invertedIndex = self.generate_inverted_index(self.docno,self.documents)

    def extract_text(self, path2docs):

        files = glob.glob(os.path.join(path2docs, '*'))
        #print(files)
        doc_numbers = list()
        text = list()
        
        for file in files:
           
            raw=[]
            try:
                with open(file, 'r', encoding='utf-8') as f:

                    content = f.read()
                    
                    #Retrieving file name
                    fname=file.split('\\')
                    fname=fname[-1]
                    doc_numbers.append(fname)

                    stripped_content = content.replace('\n', ' ') 
                    stripped_content = stripped_content .replace('TITLE:','')
                    stripped_content = stripped_content .replace('SUMMARY:','')
                    stripped_content = stripped_content .replace('DETAILED DESCRIPTION:','')
                    stripped_content = stripped_content .replace('ELIGIBILITY CRITERIA:','')
                    stripped_content = stripped_content .replace('Inclusion Criteria:','')
                    stripped_content = stripped_content .replace('Exclusion Criteria:','')
                    raw.append(content)
                    text.append(stripped_content)
                    
            except:
                continue
        text.pop()
        raw.pop()
        doc_numbers.pop()             
        return doc_numbers, text


    def preprocess(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        preprocessed = list()
        for t in text:
            t = t.lower()
            preprocessed.append(tokenizer.tokenize(t))
        return preprocessed

    def preprocess_str(self, sentence):
        tokenizer = RegexpTokenizer(r'\w+')
        sentence = tokenizer.tokenize(sentence.lower())
        return sentence

    def get_vocab(self, text):

        vocab = list()
        for doc in text:
            vocab.extend(doc)
        set(vocab)
        return set(vocab)
    
    def generate_inverted_index(self,docno,documents):
        for idx,doc in enumerate(documents):
            for token in doc:
                if token in self.invertedIndex:
                    self.invertedIndex[token].add(docno[idx])
                else:
                    self.invertedIndex[token] = {docno[idx]}
        #print(self.invertedIndex)

        with open('InvertedIndex.txt', 'w', encoding='utf-8') as f:
            for key,value in self.invertedIndex.items():

                f.write(key+':'+str(value)+'\n')
                
        return self.invertedIndex

    def idf(self, term):
        n_term = len(self.invertedIndex[term])
        if n_term == 0:
            return 0
        else:
            return math.log(self.N / n_term)

    def tf(self, term, doc):

        terms_in_doc = Counter(doc)
        max_term = max(terms_in_doc.values()) 

        return terms_in_doc[term] / max_term

    def get_vector(self, terms, document, idf_scores):

        vector = list()
        for term, idf in zip(terms, idf_scores):
            tf_idf = self.tf(term, document) * idf
            vector.append(tf_idf)

        return vector

    def similarity_scores(self, query):
        query = self.preprocess_str(query)  
        idf_scores = [self.idf(term) for term in query]
        query_vec = self.get_vector(query, query, idf_scores)
        print(idf_scores)
        print(query_vec)
        # Get similarity scores for each document
        similarity_scores = dict()
        for doc, no in zip(self.documents, self.docno): 
            doc_vec = self.get_vector(query, doc, idf_scores)
            
            # caculate the cosine similarity
            if np.dot(query_vec, doc_vec)!=0:
                cosine_sim = np.dot(query_vec, doc_vec) / \
                (np.sqrt(np.sum(np.square(query_vec))) * np.sqrt(np.sum(np.square(doc_vec))))
            else: cosine_sim = 0
            
            similarity_scores [no] = cosine_sim

        # Sorting
        similarity_scores  = sorted(similarity_scores .items(), key=operator.itemgetter(1), reverse=True)

        return similarity_scores 
    # def extract_queries(self, path2queries):
    #     queries = open(path2queries, encoding='utf-8').read()
    #     soup = BeautifulSoup(queries, 'lxml')
    #     queries = list()
    #     for q in soup.find_all('desc'):
    #         q = q.text.split()      # Get rid of 'Description:'
    #         del q[0]
    #         queries.append(' '.join(q))
    #     return queries

def start(query):
    sys.stdout.reconfigure(encoding='utf-8')
    folder_path = 'C:\\Users\\91949\\OneDrive\\Documents\\Courses\\Information Retrieval\\Project\\Medical Information Retrieval\\docs\\'
    path2docs = folder_path

    articles = IRModel(path2docs)
    queries = [query]
    

    # Rank total documents with the cosine similarity and tf-idf
    res=[]
    for q in queries:
        print(q)
        sim_scores = articles.similarity_scores(q)
        docno = [no for no, score in sim_scores]
        top_10_docs= docno[:10]
        res=top_10_docs
        print("Top 10 Documents")
        print(top_10_docs)
        print('Documents and similarity score')
        print(sim_scores[:10])
    return res



        
        




    

    


    

   
