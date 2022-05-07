# Wikipedia Toolkits

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