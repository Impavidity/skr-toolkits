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
from skr2.utils import crash_on_ipy

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
  parser.add_argument("--input_file_path", type=str)
  parser.add_argument("--output_file_path", type=str)
  args = parser.parse_args()

  os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
  fout = open(args.output_file_path, 'w')
  with open(args.input_file_path) as fin:
    for line in tqdm(fin):
      processed_ex = process_example(json.loads(line))
      if processed_ex:
        fout.write(json.dumps(processed_ex) + "\n")
