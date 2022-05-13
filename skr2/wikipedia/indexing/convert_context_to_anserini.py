import argparse
import os
import json
from tqdm import tqdm

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file", type=str)
  parser.add_argument("--output_file", type=str)
  args = parser.parse_args()

  fin = open(args.input_file)
  os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
  fout = open(args.output_file, 'w')
  for line in tqdm(fin):
    ex = json.loads(line)
    title = ex["title"].replace(" - Wikipedia", "")
    fout.write(json.dumps({
      "id": "{}-{}-{}-{}".format(title, ex["para_id"], ex["split_id"], ex["sent_id"]),
      "contents": "{}. {}".format(title, ex["sent"])
    }) + "\n")