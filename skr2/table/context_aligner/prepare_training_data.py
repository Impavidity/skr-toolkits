"""
Dataset Format of Tevatron
{
   "query_id": "<query id>",
   "query": "<query text>",
   "positive_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
   ],
   "negative_passages": [
     {"docid": "<passage id>", "title": "<passage title>", "text": "<passage body>"}
   ]
}
"""
import json
from skr2.structure.context_table_pair import ContextTablePair
from skr2.structure.table import table_linearization, TableConfig
from skr2.wikipedia.wiki_extractor import WikiExtractor
from skr2.structure.context import ContextConfig
from pyserini.search.lucene import LuceneSearcher
from fuzzywuzzy import fuzz
from skr2.utils import crash_on_ipy

def process_example_3(ex, extractor: WikiExtractor, searcher: LuceneSearcher):
  # Re-align the text with table with our toolkits (for negative alignment/ introducing negative signal)
  # Extract facts
  # Aligning with Text (I wrote this script before)
  # Yes or No; How to have negative example?
  table_config = TableConfig(no_process=True)
  pair = ContextTablePair.from_totto_annotation(ex, table_config=table_config)

def process_example_2(ex, extractor: WikiExtractor, searcher: LuceneSearcher):
  # Re-align the text with table with our toolkits (for negative alignment/ introducing negative signal)
  # Extract facts
  # Aligning with Text (I wrote this script before)
  # Yes or No; How to have negative example?
  table_config = TableConfig(no_process=True)
  pair = ContextTablePair.from_totto_annotation(ex, table_config=table_config)
  # matched_sentence = extractor._matching(pair.table, [pair.context.meta["original_sentence"]], force_valid=True)[0]
  # extract facts: Extract the lines having elements matched.
  # for highlighted_cell in matched_sentence["highlighted_cells"]:
  #   query.append(highlighted_cell["text"])
  #   if highlighted_cell["row_idx"] not in row_ids:
  #     row_ids.append(highlighted_cell["row_idx"])
  # if 0 not in row_ids:
  #   row_ids.append(0)

  # How to solve the false negative?

  row_ids = []
  query = []
  for highlighted_cell in pair.context.annotations:
    query.append(pair.table.get_cell(highlighted_cell[0], highlighted_cell[1]).text)
    if highlighted_cell[0] not in row_ids:
      row_ids.append(highlighted_cell[0])
  if 0 not in row_ids:
    row_ids.append(0)
  row_ids = sorted(row_ids)
  extracted_facts = [pair.table.table[row_id] for row_id in row_ids]
  table_str = table_linearization(extracted_facts)
  print(" ".join(query))
  hits = searcher.search(" ".join(query))
  # count = 0
  for i in range(0, min(3, len(hits))):
    sent = json.loads(searcher.doc(hits[i].docid).raw())["contents"].split(". ", 1)[1]
    if fuzz.partial_ratio(sent, pair.context.meta["original_sentence"]) < 85:
      yield {
        "table": table_str,
        "text": sent,
        "label": 0
      }
      # count += 1
      # if count >= 3:
      #   break

  yield {
    "table": table_str,
    "text": pair.context.meta["original_sentence"],
    "label": 1
  }

def process_example(ex):
  # linearize table
  # query is the table
  # doc is the sentences
  table_config = TableConfig(
    max_row_num=15,
    ignore_hierarchical_schema_table=False,
    ignore_empty_column_table=True,
    infer_column_type=False,
    no_process=True)
  pair = ContextTablePair.from_totto_annotation(ex, table_config=table_config)
  if pair.table.is_valid():
    table_str = table_linearization(pair.table.linearize_table(use_column_name=True))
    ctx_str = pair.context.text
    return {
      "query_id": str(ex["example_id"]),
      "query": table_str,
      "positive_passages": [
        {"docid": str(ex["example_id"]), "title": ex["table_page_title"], "text": ctx_str}
      ],
      "negative_passages": []
    }
  else:
    return None

if __name__ == "__main__":
  import os
  import argparse
  from tqdm import tqdm
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type=str)
  parser.add_argument("--output_file", type=str)
  args = parser.parse_args()
  exs = []
  extractor = WikiExtractor("EXTRACT_TABLE_HTML", None, None, None)
  from pyserini.util import download_and_unpack_index
  local_filename = "lucene-index.enwiki-html-20211013-sentences"
  # dir = download_and_unpack_index(url, local_filename=local_filename, prebuilt=True)
  searcher = LuceneSearcher("data/wikipedia/en/context/index/")

  os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
  fout = open(args.output_file, 'w')

  with open(args.input_file) as fin:
    for line in tqdm(fin):
      # processed_ex = process_example(json.loads(line))
      # process_example_3(json.loads(line), extractor, searcher)
      for example in process_example_2(json.loads(line), extractor, searcher):
        fout.write(json.dumps(example) + "\n")
      # raise NotImplementedError()

