# Wikipedia Toolkit Usages

## Setup
```
pip install wikiextractor
```

## How to crawl Wikipedia HTML pages?

First we need to obtain the URLs of Wikipedia pages.
Download Wikipedia `pages-articles.xml.bz2` dumps from 
[official dump](https://dumps.wikimedia.org/backup-index.html).

Then we can extract the URLs from the dump. 
Here we use Chinese Wikipedia as examples.
We download the dump `zhwiki-20210701-pages-articles.xml.bz2` in `dumps` directory.
```
mkdir -p dumps
# Download the dump into the directory.

nohup python -m wikiextractor.WikiExtractor \
dumps/zhwiki-20210701-pages-articles.xml.bz2 \
-o dumps/zh_text  --processes 96 > enwiki-20210701-pages-articles.extract.log &

mkdir -p data/examples/wikipedia/zh
python skr2/wikipedia/extract_wiki_urls.py \
--input_path dumps/zh_text \
--language zh \
--output_path data/examples/wikipedia/zh/urls.txt
```

Then we can crawl the wikipedia HTML pages based on the extracted URLs.
You can decide the number of chunks you want to have.
Here we set the chunk size as 150.
```
mkdir -p data/examples/wikipedia/zh/html/
mkdir -p logs/wikipedia/zh/

for i in {0..149}; do
  seg_id=${i}
  echo ${seg_id}
  python -u skr2/wikipedia/page_crawler.py \
  --url_file  data/examples/wikipedia/zh/urls.txt \
  --seg_id ${seg_id} --segments 150 \
  --revisit logs/wikipedia/zh/revisit_${seg_id}.jsonl \
  --output_file data/examples/wikipedia/zh/html/seg_${seg_id}.jsonl > logs/wikipedia/zh/seg_${seg_id}.log
done
```

## How to index Wikipedia Sentences?

Extract context in sentence level.
```
nohup python -u -m skr2.wikipedia.wiki_extractor \
--input_file path_to_wikipedia_html \
--sentence_output_file data/wikipedia/en/context/sentence.json \
--mode EXTRACT_CONTEXT > extract_context.log &
```

Then convert the output into [Anserini](https://github.com/castorini/anserini) format.

```
nohup python -u skr2/wikipedia/indexing/convert_context_to_anserini.py \
--input_file data/wikipedia/en/context/sentence.json \
--output_file data/wikipedia/en/context/anserini/1.json > convert_context.log &
```

Indexing.

```
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input data/wikipedia/en/context/anserini/ \
  --index data/wikipedia/en/context/index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```