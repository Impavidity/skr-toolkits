import argparse
import os
import json
import random
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import querybuilder
# from skr2.utils import crash_on_ipy
from tqdm import tqdm
from thefuzz import fuzz
random.seed(1234)


must = querybuilder.JBooleanClauseOccur["must"].value
should = querybuilder.JBooleanClauseOccur['should'].value
IGNORE = ["family name"]

def exact_match_query_compose(query):
    terms = []
    for term in query.split():
        try:
            terms.append(querybuilder.get_term_query(term))
        except:
            print("Ignore {}".format(term))
    boolean_query_builder = querybuilder.get_boolean_query_builder()
    for term in terms:
        boolean_query_builder.add(term, must)

    query = boolean_query_builder.build()
    return query

class TripleQueryComposer:
    def __init__(self):
        pass

    def exact_match_query(self, triple):
        subj, _, obj = triple
        return exact_match_query_compose(subj + " " + obj)

    def default_query(self, triple):
        subj, _, obj = triple
        return subj + " " + obj


class Retriever:
    def __init__(self, index_paths) -> None:
        self.searcher = {}
        for index_path in index_paths:
            self.searcher[index_path] = LuceneSearcher(index_path)
            print("Build Search on Index {}".format(index_path))
    
    def search(self, query, topk=10):
        results = {}
        for index_path in self.searcher:
            # print("Search on Index {}".format(index_path))
            searcher = self.searcher[index_path]
            hits = searcher.search(query, k=topk)
            # print("Done the search")
            results[index_path] = hits
        return results
    
    def get_doc(self, index_path, doc_id):
        return json.loads(self.searcher[index_path].doc(doc_id).raw())

class NegativeSampler:
    def __init__(self, index_path):
        self.index_path = index_path.split(",")
        self.retriever = Retriever(self.index_path)
        self.query_composer = TripleQueryComposer()
        print("Build Search on Index {}".format(index_path))
        

    def retrieve(self, triple, topk=10):
        query = self.query_composer.exact_match_query(triple)
        results = self.retriever.search(query, topk=topk)
        for index_path in results:
            for item in results[index_path]:
                sentence = self.retriever.get_doc(index_path=index_path, doc_id=item.docid)
                print(sentence)

if __name__ == "__main__":
    sampler = NegativeSampler(index_path="../indexes/en-wiki-sentences-index/index")
    sampler.retrieve(triple=["Matthew Nuthall", "place of birth", "Pontypridd"])
    sampler.retrieve(triple=["Matthew Nuthall", "date of birth", "1983"])
