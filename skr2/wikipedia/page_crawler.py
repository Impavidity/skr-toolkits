from multiprocessing import Pool, Value, Lock
import urllib3
http = urllib3.PoolManager()
urllib3.disable_warnings()
import json
import time
import datetime

class AtomicCounter(object):
  def __init__(self):
    self.value = Value("i", 0)
    self.lock = Lock()

  def increment(self):
    with self.lock:
      self.value.value += 1
      return self.value.value
counter = AtomicCounter()


def scrape(url):
  count = 0
  v = counter.increment()
  if v % 500 == 0:
    print("[{}] {} examples processed.".format(datetime.datetime.now(), v))

  while True:
    if count > 10:
      break
    r = http.request('GET', url)
    if r.status == 200:
      data = r.data.decode('utf-8')
      return data
    elif r.status == 429:
      time.sleep(1)
      count += 1
    elif r.status == 404 or r.status == 400:
      return ""
  return {
    "url": url,
    "status": 429,
  }

def chunks(l, n):
  """Yield n number of striped chunks from l."""
  for i in range(0, n):
    yield l[i::n]

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--url_file", type=str, required=True)
  parser.add_argument("--seg_id", type=int)
  parser.add_argument("--segments", type=int)
  parser.add_argument("--revisit", type=str)
  parser.add_argument("--output_file", type=str)

  args = parser.parse_args()
  urls = []
  with open(args.url_file) as fin:
    for line in fin:
      urls.append(line.strip())
  frevisit = open(args.revisit, "w")
  fout = open(args.output_file, "w")
  segments = list(chunks(urls, args.segments))
  urls_segment = segments[args.seg_id]
  p = Pool(10)
  data = p.map(scrape, urls_segment)
  assert len(urls_segment) == len(data)
  for url, item in zip(urls_segment, data):
    if isinstance(item, dict):
      frevisit.write(json.dumps(item) + "\n")
    elif item == "":
      continue
    else:
      fout.write(json.dumps({
        "url": url,
        "data": item
      }) + "\n")