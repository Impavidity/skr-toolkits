import argparse
import os
import json
import random
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import querybuilder
from skr2.utils import crash_on_ipy
from tqdm import tqdm
from thefuzz import fuzz
random.seed(1234)


must = querybuilder.JBooleanClauseOccur["must"].value
should = querybuilder.JBooleanClauseOccur['should'].value
IGNORE = ["family name"]
def query_compose(query):
    # terms = []
    # for term in query.split():
    #     try:
    #         terms.append(querybuilder.get_term_query(term))
    #     except:
    #         print("Ignore {}".format(term))
    # boolean_query_builder = querybuilder.get_boolean_query_builder()
    # for term in terms:
    #     boolean_query_builder.add(term, must)

    # query = boolean_query_builder.build()
    # return query
    return query

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="web_nlg")
    parser.add_argument("--dataset_name", type=str, default="web_nlg")
    parser.add_argument("--dataset_config_name", type=str, default="release_v3.0_en")
    parser.add_argument("--index_paths", type=str)
    parser.add_argument("--local_file_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    if args.dataset == "local":
        assert args.local_file_path is not None
        dataset = {} # q -> [{}]
        with open(args.local_file_path) as fin:
            for line in fin:
                # group into example level
                ex = json.loads(line)
                if ex["sentence"] not in dataset:
                    dataset[ex["sentence"]] = []
                dataset[ex["sentence"]].append({
                    "triple": ex["triple"].split("<S>")[1:],
                    "label": ex["label"]
                })
        dataset = list(dataset.items())
        random.shuffle(dataset)
        train_dataset = dataset[:int(len(dataset) * 0.8)]
        dev_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset))]
        os.makedirs(args.output_dir, exist_ok=True)
        for split, name in zip([train_dataset, dev_dataset], ["train", "validation"]):
            with open(os.path.join(args.output_dir, "{}.json".format(name)), "w") as fout:
                for example in split:
                    for anno in example[1]:
                        fout.write(json.dumps({
                            "sentence": example[0],
                            "triple": anno["triple"],
                            "label": anno["label"]
                        }) + "\n")
            

    elif args.dataset == "filtered_positive":
        fout = open(os.path.join(args.output_dir, "pseudo-train.jsonl"), "w")
        retriever = Retriever(args.index_paths.split(","))
        with open(args.local_file_path) as fin:
            for line in tqdm(fin):
                ex = json.loads(line)
                triples = ex["triples"].split("<sep>")
                for triple in triples:
                    subj = triple.split("<R>")[0].split("<H>")[1]
                    obj = triple.split("<T>")[1]
                    pred = triple.split("<R>")[1].split("<T>")[0]
                    fout.write(json.dumps({
                        "text": ex["sentence"],
                        "triple": " ".join([subj, pred, obj]),
                        "label": 0
                    }) + "\n")
                    if pred in IGNORE:
                        continue
                    query = query_compose(subj + " " + obj)
                    results = retriever.search(query, topk=5)
                    for index_path in results:
                        for result in results[index_path]:
                            doc = retriever.get_doc(index_path, result.docid)
                            text = doc["contents"].split(". ", 1)[1]
                            if fuzz.partial_ratio(text, ex["sentence"]) < 85:
                                fout.write(json.dumps({
                                    "text": text,
                                    "triple": " ".join([subj, pred, obj]),
                                    "label": 1
                                }) + "\n")
                                
