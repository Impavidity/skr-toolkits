import argparse
import glob
from tqdm import tqdm
import xml.etree.ElementTree as ET

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--input_path", type=str)
  parser.add_argument("--language", type=str)
  parser.add_argument("--output_path", type=str)

  args = parser.parse_args()

  files = list(sorted(glob.glob(args.input_path + '/*/*', recursive=True)))

  urls = []
  for file in tqdm(files):
    docs = []
    with open(file) as fin:
      doc_str = []
      in_doc = False
      for line in fin:
        if line.startswith("<doc"):
          in_doc = True
          doc_str.append(line.strip())
        elif line.startswith("</doc"):
          doc_str.append(line)
          if len(doc_str) > 3:
            # It only has title
            docs.append(doc_str)
          doc_str = []
        else:
          line = line.strip()
          if len(line) > 0:
            doc_str.append(line.strip())
    for doc in docs:
      d = ET.fromstring(doc[0] + "</doc>")
      urls.append(d.attrib["url"])

  with open(args.output_path, "w") as fout:
    for url in urls:
      fout.write(url + "\n")