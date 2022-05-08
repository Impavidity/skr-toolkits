"""
Instructions:
1. To debug the code, switch to `debug_page` in the main function.
2. Language supports:
  [] English
  [] Chinese
  [] Vietnamese
  How to select language: TableExtractor has `language` argument.
3. The extraction modes are
  - EXTRACT_TABLE_HTML: This is for raw input for indexing; structural encoding can be used. The most
                        simple way to index is to linearization via `html_table_to_json.py`
  - EXTRACT_TABLE: We currently not support this mode; because the tables can be obtained via
                  EXTRACT_TABLE_HTML + `html_table_to_json`
  - EXTRACT_CONTEXT_TABLE_PAIR: Extract the sentence and table pairs from HTML.
                                The table is stored as list. Because alignment are derived so original HTML format is not used.
                                The sentence is text. The alignments should be stored. The table info is a reference.
                                Two files will be stored: table.json and sentence.json.
4. The main function is `_extract_table`

"""

import multiprocessing
from bs4 import BeautifulSoup
import unicodedata
from argparse import ArgumentParser
from pathlib import Path
import sys
from tqdm import tqdm
import json
import os
import traceback
from nltk import sent_tokenize
import copy
import re
from collections import defaultdict
import nltk.corpus
import string
import spacy
from skr2.utils import crash_on_ipy

LANGUAGE_TO_TITLE = {
  "en": " - Wikipedia",
}

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)

def clean_cell_value(cell_val):
  val = unicodedata.normalize('NFKD', cell_val)
  # val = val.encode('ascii', errors='ignore')
  # val = str(val, encoding='ascii')
  return val

def get_cell_text(cell):
  text = []
  for content in cell.contents:
    s = content.string
    if s is not None:
      s = s.strip()
      if s and not s.startswith(".mw-parser-output"): text.append(clean_cell_value(s))
    else:
      # We add flagicon span
      if content.name == "span":
        possible_span_text = content.text.strip()
        if possible_span_text:
          text.append(clean_cell_value(possible_span_text))
        elif "class" in content.attrs:
          if "flagicon" in content["class"]:
            link = content.find("a")
            if link:
              text.append(clean_cell_value(link["title"].strip()))
        elif "title" in content.attrs:
          text.append(clean_cell_value(content["title"]))
        elif "data-sort-value" in content.attrs:
          try:
            val = content["data-sort-value"].replace("!", "").strip()
            text.append(str(int(val)))
          except:
            print("invalid data-sort-value", content)
        elif len(content.attrs) == 0:
          continue
        elif len(content.attrs) == 1 and "style" in content.attrs:
          continue
        else:
          print(content)
          # raise NotImplementedError()
      elif content.name == "div":
        recursive_text = get_cell_text(content)
        if recursive_text:
          text.append(recursive_text)
      else:
        try:
          v = content.text
          if isinstance(v, str):
            text.append(clean_cell_value(v.strip()))
        except:
          print("invalid content", content)
  return " ".join(text)

def get_section_titles(table):
  if table.name == "div" and table.attrs.get("id", None) == "bodyContent":
    return []
  section_titles = []
  headers = []
  for item in table.previous_siblings:
    if item.name and len(item.name) == 2 and item.name.startswith("h") and item.name[1].isdigit() and item.name not in headers:
      header_num = int(item.name[1])
      if len(headers) == 0 or (len(headers) > 0 and header_num < int(headers[-1][1])):
        headers.append(item.name)
        head = item.find("span", class_="mw-headline")
        section_titles.append(head.text)
      if header_num == 2:
        break
  if len(headers) == 0:
    section_titles = get_section_titles(table.parent)
  return section_titles[::-1]

def get_table_caption(table):
  caption = None if table.caption is None else table.caption.text.strip()
  if caption is None:
    if table.parent.name == "td":
      # It is subtable, try to find caption
      for item in table.previous_siblings:
        if item.name == "b": # bold text as caption
          return item.text
  else:
    return caption

def is_valid_cell(value):
  if value in ["", "-"]:
    return False
  return True

def is_num(value):
  try:
    float(value)
    return True
  except:
    return False

NUMERICAL_NER_TAGS = {'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL'}
IGNORE_TOKENS = ['(', ')']
SPECIAL_TOKENS = ['°C', '°F']
def is_valid_statistics(sample_value_entry):
  if sample_value_entry.get("header", None) == "Year":
    return False
  value = sample_value_entry["text"]
  if "," in value or "." in value or ":" in value:
    if is_num(value.replace(",", "").replace(".", "").replace(":", "")):
      return True

  if any([special in value for special in SPECIAL_TOKENS]):
    return True

  if len(value) == 4 or len(value) == 1:
    return False

  num_numerical_tags = 0
  num_other_tags = 0
  for token, tag in zip(sample_value_entry["tokens"], sample_value_entry["ner_tags"]):
    if tag in NUMERICAL_NER_TAGS:
      num_numerical_tags += 1
    elif token in IGNORE_TOKENS:
      continue
    else:
      num_other_tags += 1
  return True if num_numerical_tags > num_other_tags else False


# We use Non-date NER tags
def infer_column_type_from_sampled_value(sample_value_entry):
    if not 'ner_tags' in sample_value_entry:
        return 'text'

    if sample_value_entry["header"] == "Year":
      return 'text'

    num_numerical_tags = 0
    num_other_tags = 0

    for tag in sample_value_entry['ner_tags']:
        if tag in NUMERICAL_NER_TAGS:
            num_numerical_tags += 1
        else:
            num_other_tags += 1

    return 'real' if num_numerical_tags >= num_other_tags else 'text'

def normalize_cell_value(text):
  for month in [("Jan ", "January "),
                ("Feb ", "February "),
                ("Mar ", "March "),
                ("Apr ", "April "),
                ("May ", "May "),
                ("Jun ", "June "),
                ("Jul ", "July "),
                ("Aug ", "August "),
                ("Sep ", "September "),
                ("Oct ", "October "),
                ("Nov ", "November "),
                ("Dec ", "December ")]:
    if month[0] in text:
      text = text.replace(month[0], month[1])
  return text

def exact_match(x_list, y_list):
  x_str = " ".join(x_list)
  y_str = " ".join(y_list)
  # Normalize date

  if x_str == y_str:
    return True
  else:
    return False

def is_invalid_exact_match(element, alignment, annotated_sentence):
  if element["is_stat"]:
    for i in range(alignment["start_idx"], alignment["end_idx"]):
      if annotated_sentence["ner_tags"][i] in ["DATE"]:
        return True
  return False

def partial_match(x_list, y_list):
  x_str = " ".join(x_list)
  y_str = " ".join(y_list)
  if x_str in STOPWORDS or x_str in PUNKS:
    return False
  if re.search(rf"\b{re.escape(x_str)}\b", y_str):
    assert x_str in y_str
    return True
  else:
    return False

class HtmlTable(object):
  def __init__(self, table):
    self.table = table
    self.page_title = None
    self.caption = None if table.caption is None else table.caption.text.strip()
    if self.caption is None:
      self.caption = get_table_caption(table)
    if self.caption:
      pass
      # TODO: clean the caption
      # self.caption = clean_cell_value(self.caption)

    self.remove_hidden()
    self.normalize_table(deep=True)
    self.get_cells()
    self.section_titles = get_section_titles(self.table)

  def get_int(self, cell, key):
    try:
      return int(cell.get(key, 1))
    except ValueError:
      try:
        return int(re.search('[0-9]+', cell[key]).group())
      except:
        return 1

  def get_cloned_cell(self, cell, rowspan=1, deep=False):
    if deep:
      # Hacky but works
      return BeautifulSoup(str(cell), 'html.parser').contents[0]
    tag = BeautifulSoup().new_tag(cell.name)
    if rowspan > 1:
      tag['rowspan'] = rowspan
    return tag

  def normalize_table(self, deep=False):
    """Fix the table in-place"""
    num_cols = 0
    for tr in self.table.find_all('tr', recursive=True):
      for cell in tr.find_all(['th', 'td'], recursive=True):
        colspan = self.get_int(cell, 'colspan')
        rowspan = self.get_int(cell, 'rowspan')
        if colspan <= 1:
          continue
        cell['old-colspan'] = cell['colspan']
        del cell['colspan']
        for i in range(2, colspan + 1):
          cell.insert_after(self.get_cloned_cell(cell, rowspan=rowspan, deep=deep))
      num_cols = max(num_cols, len(tr.find_all(['th', 'td'], recursive=True)))
    counts = defaultdict(int)
    spanned_cells = dict()
    for row_id, tr in enumerate(self.table.find_all('tr', recursive=True)):
      cell = None
      cells = tr.find_all(['th', 'td'], recursive=True)
      k = 0
      for i in range(num_cols):
        if counts[i] > 0:
          # Create a new element caused by rowspan
          new_cell = self.get_cloned_cell(spanned_cells[i], deep=deep)
          if not cell:
            tr.insert(0, new_cell)
          else:
            cell.insert_after(new_cell)
          cell = new_cell
          counts[i] -= 1
        else:
          if k >= len(cells):  # Unfilled row
            continue
          cell = cells[k]  # The cell value
          k += 1
          rowspan = self.get_int(cell, 'rowspan')
          if rowspan <= 1:
            continue
          counts[i] = rowspan - 1  # How many cells should be appended for this column in the following rows
          spanned_cells[i] = cell
          cell['old-rowspan'] = cell['rowspan']
          del cell['rowspan']


  def get_cells(self):
    # Get all cells
    self.cells = []
    self.rows = []
    for x in self.table.find_all('tr'):
      row = []
      for y in x.find_all(['th', 'td']):
        cell_text = get_cell_text(y)
        self.cells.append(cell_text)
        row.append(cell_text)
      self.rows.append(row)

  def annotate_cell_values(self, nlp):
    self.annotations = []
    schema = {}
    if len(self.rows) > 0:
      for col_idx, cell_value in enumerate(self.rows[0]):
        schema[col_idx] = cell_value
    for row_idx, row in enumerate(self.rows):
      annotation = []
      for col_idx, cell_value in enumerate(row):
        cell_value = normalize_cell_value(cell_value)
        text = nlp(cell_value)
        annotated_cell_value = {
          "row_idx": row_idx,
          "col_idx": col_idx,
          "ner_tags": [token.ent_type_ for token in text],
          "tokens": [token.text for token in text],
          "text": cell_value,
          "header": schema.get(col_idx, "None")}
        # cell_value_type = infer_column_type_from_sampled_value(annotated_cell_value)
        annotated_cell_value["is_stat"] = is_valid_statistics(annotated_cell_value)
        annotation.append(annotated_cell_value)
      self.annotations.append(annotation)

  def check_hidden(self, tag):
    classes = tag.get('class', [])
    if 'reference' in classes or 'sortkey' in classes:
      return True
    if 'display:none' in tag.get('style', ''):
      return True
    return False

  def remove_hidden(self):
    """Remove hidden elements."""
    for tag in self.table.find_all(self.check_hidden):
      tag.extract()


class TableExtractor(multiprocessing.Process):
  def __init__(self, mode, job_queue, table_queue, sentence_queue, language="en", **kwargs):
    super(TableExtractor, self).__init__(**kwargs)
    self.mode = mode
    self.job_queue = job_queue
    self.table_queue = table_queue
    self.sentence_queue = sentence_queue
    # self.example_queue = example_queue
    self.language = language
    self.nlp = spacy.load('en_core_web_sm')
    # print('loaded spacy model')

  def run(self):
    job = self.job_queue.get()
    while job is not None:
      ex = json.loads(job)
      url = ex["url"]
      page = ex["data"]
      try:
        # context_sents = self.extract_context(url, page)
        if self.mode == "EXTRACT_TABLE_HTML":
          for table in self._extract_table(url, page):
            self.table_queue.put(table)
        # if self.mode == ""

        # for table in self._extract_table(url, page):
        #   self.table_queue.put(table)
          # matched_sentences = self.matching(table, context_sents)
          # for matched_sentence in matched_sentences:
          #   self.sentence_queue.put(matched_sentence)

      except:
        typ, value, tb = sys.exc_info()
        print('*' * 30 + 'Exception' + '*' * 30, file=sys.stderr)
        print(f'url={url}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
      job = self.job_queue.get()

  def extract_table_html(self, url, page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    page_title = soup.title.text.replace(LANGUAGE_TO_TITLE[self.language], "")
    rs = soup.find_all("")# soup.find_all(class_='wikitable') + soup.find_all(class_="infobox") # + soup.find_all(class_='wikitable')
    for idx, r in enumerate(rs):
      try:
        # table = HtmlTable(r)
        # table.page_title = page_title
        example = {
          "url": url,
          "idx": idx,
          "html": str(r),
          "page_title": page_title,
          "section_titles": get_section_titles(r),
          "table_caption": get_table_caption(r),
        }
      except:
        typ, value, tb = sys.exc_info()
        print('*' * 30 + 'Exception' + '*' * 30, file=sys.stderr)
        print(f'url={url}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        continue
      yield example

  def extract_table(self, url, page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    page_title = soup.title.text.replace(LANGUAGE_TO_TITLE[self.language], "")
    rs = soup.find_all(class_='wikitable')
    for idx, r in enumerate(rs):
      table = HtmlTable(r)
      table.annotate_cell_values(self.nlp)
      table.page_title = page_title
      title_text = self.nlp(table.page_title)
      title_element = {
          "row_idx": -1,
          "col_idx": -1,
          "ner_tags": [token.ent_type_ for token in title_text],
          "tokens": [token.text for token in title_text],
          "text": page_title,
          "is_stat": False,
          "header": "None"}
      example = {
        "url": url,
        "idx": idx,
        "annotations": table.annotations,
        "page_title": title_element,
        "section_titles": table.section_titles,
        "table_caption": table.caption
      }
      yield example

  def _extract_table(self, url, page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    page_title = soup.title.text # .replace(" - Wikipedia", "")
    rs = soup.find_all(class_='wikitable') # + soup.find_all(class_="infobox") # + soup.find_all(class_='wikitable')
    # rs = soup.find_all(lambda tag: tag.name == 'table' and not tag.find_parent('table'))
    for idx, r in enumerate(rs):
      # print(r)
      table = HtmlTable(r)
      table.page_title = page_title
      example = {
        "url": url,
        "idx": idx,
        # "html": str(r),
        "annotations": table.rows,
        "page_title": page_title,
        "section_titles": table.section_titles,
        "table_caption": table.caption
      }
      yield example

  def extract_interlingual_link(self, url, page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    raise NotImplementedError()

  def extract_infobox(self, url, page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    rs = soup.find_all(class_='infobox')

  def extract_context(self, url, page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    # Remove sup
    # Simple superscript extraction
    for element in soup.find_all('sup'):
      element.extract()

    # More complex superscript extraction for this example:
    for element in soup.find_all(lambda e: e and e.name == 'font' and e.has_attr('style') and
                                           'position:relative' in e['style'] and
                                           'top:' in e['style']):
      element.extract()
    paras = soup.find_all("p")
    sentences = []
    for paragraph in paras:
      for para in paragraph.text.split("\n"):
        sents = sent_tokenize(para)
        sentences.extend(sents)
    return sentences

  def matching(self, table, sentences):
    # Follow the ToTTo dataset construction
    # 1. Number Matching: Non-date number of at least 3 non-zero digits.
    # number_candidates = []
    # for row in table["annotations"]:
    #   # For each row, we try to match digits
    #   for annotated_cell_value in row:
    #     if is_valid_statistics(annotated_cell_value)
    #       # Try to match the sentences candidates
    #       number_candidates.append(annotated_cell_value)
    # #
    candidate_sentences = []
    # for sentence in sentences:
    #   highlighted_cells = []
    #   for number_candidate in number_candidates:
    #     if number_candidate["text"] in sentence:
    #       highlighted_cells.append(number_candidate)
    #   if len(highlighted_cells) > 0:
    #     candidate_sentences.append({
    #       "sentence": annotated_sentence,
    #       "highlighted_cells": highlighted_cells
    #     })

    # 2. Row matching: matching at least 3 elements from the same raw or meta data.

    # def deduplicate(highlighted_cells, annotated_sentence):
    #   sentence_token_bio = ["O"] * len(annotated_sentence["tokens"])
    #   deduplicated_highlighted_cells = []
    #   for highlighted_cell in highlighted_cells:
    #     deduplicated_alignment = []
    #     for alignment in highlighted_cell["alignment"]:
    #       is_duplicate = False
    #
    #       for i in range(alignment["start_idx"], alignment["end_idx"]):
    #         if sentence_token_bio[i] != "O":
    #           is_duplicate = True
    #         else:
    #           sentence_token_bio[i] = "I"
    #       if not is_duplicate:
    #         deduplicated_alignment.append(alignment)
    #         break #
    #     if len(deduplicated_alignment) > 0:
    #       pass
    #
    #   return deduplicated_highlighted_cells

    def get_highlighted_spans(highlighted_cells, annotated_sentence):
      spans = set()
      for highlighted_cell in highlighted_cells:
        spans.add(highlighted_cell["text"])
      return spans

    def is_valid_sentence_candidate(highlighted_cells, annotated_sentence):
      # has stat candidates:
      for highlighted_cell in highlighted_cells:
        # if is_valid_statistics(highlighted_cell):
        if highlighted_cell["is_stat"]:
          return True
      highlighted_spans = get_highlighted_spans(highlighted_cells, annotated_sentence)
      if len(highlighted_spans) >= 2:
        return True
      return False

    for sentence in sentences:
      try:
        highlighted_cells = []
        annotated_sentence = None
        for row in table["annotations"]:
        # For each row, we try to match elements
          cells, annotated_sentence = self.row_sentence_matching(sentence, row,
                                                                 annotated_sentence_=annotated_sentence,
                                                                 )
          highlighted_cells.extend(cells)
        # print(highlighted_cells)
        if annotated_sentence:
          aligned_element = self.element_sentence_matching(table["page_title"], annotated_sentence)
          if aligned_element is not None:
            highlighted_cells.append(aligned_element)
          # print(highlighted_cells)
        # if len(highlighted_cells) > 0:
        #   # The annotated_sentence must be not None
        #   deduplicated_highlighted_cells = deduplicate(highlighted_cells, annotated_sentence)
        # else:
        #   deduplicated_highlighted_cells = highlighted_cells
        # print(highlighted_cells)

        if is_valid_sentence_candidate(highlighted_cells, annotated_sentence):
          candidate_sentences.append({
            "url": table["url"],
            "idx": table["idx"],
            "sentence": annotated_sentence,
            "highlighted_cells": highlighted_cells
          })
      except:
        typ, value, tb = sys.exc_info()
        print('*' * 30 + 'Exception' + '*' * 30, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    return candidate_sentences

  def element_sentence_matching(self, element, annotated_sentence):
    # Doing sentence process
    # print(element["text"])
    element_tokens = element["tokens"]
    sentence_tokens = annotated_sentence["tokens"]
    require_exact = element["text"].isdigit() # any([x.isdigit() for x in element["text"]])
    # sentence might use entity alias.
    # So for each 1-n grams of the sentence, we match it with full element.

    # Let's consider exact match case first.
    alignments = []
    n = len(element_tokens)
    for i in range(len(sentence_tokens) - n + 1):
      n_gram_list = sentence_tokens[i:i + n]
      n_gram = " ".join(n_gram_list)
      if len(n_gram.strip()) == 0:
        continue
      if exact_match(n_gram_list, element_tokens):
        # special case: 20 matches May 20.
        alignment = {
          "start_idx": i,
          "end_idx": i + n,
          "tokens": n_gram_list,
          "match_type": "EXACT"
        }
        if is_invalid_exact_match(element, alignment, annotated_sentence):
          continue
        alignments.append({
          "start_idx": i,
          "end_idx": i + n,
          "tokens": n_gram_list,
          "match_type": "EXACT"
        })
    if len(alignments) == 0 and not require_exact:
      # Try the partial match
      found = False

      while n > 1 and not found:
        n -= 1
        for i in range(len(sentence_tokens) - n + 1):
          n_gram_list = sentence_tokens[i:i + n]
          # print("n gram", " ".join(n_gram_list))
          # print("element", " ".join(element_tokens))
          n_gram = " ".join(n_gram_list)
          if len(n_gram.strip()) == 0:
            continue
          if partial_match(n_gram_list, element_tokens):
            alignments.append({
              "start_idx": i,
              "end_idx": i + n,
              "tokens": n_gram_list,
              "match_type": "PARTIAL"
            })
            found = True
            break
    if len(alignments) > 0:
      element_ = copy.deepcopy(element)
      element_["alignment"] = alignments
      return element_
    return None



  def row_sentence_matching(self, sentence, row, annotated_sentence_=None, meta=None):
    # We continue to do fine-grained matching iff one exact match of the row satisfied.
    if annotated_sentence_ is not None:
      sentence = annotated_sentence_["text"]
      annotated_sentence = annotated_sentence_
    else:
      sentence = clean_cell_value(sentence)
      annotated_sentence = None
    highlighted_cells = []
    do_fine_grained_match = False
    for element in row:
      if is_valid_cell(element["text"]) and element["text"] in sentence:
        do_fine_grained_match = True
    if do_fine_grained_match:

      if annotated_sentence is None:
        text = self.nlp(sentence)
        annotated_sentence = {
          "tokens": [token.text for token in text],
          "ner_tags": [token.ent_type_ for token in text],
          "text": sentence
        }
      for element in row:
        # print(element)
        aligned_element = self.element_sentence_matching(element, annotated_sentence)
        if aligned_element is not None:
          highlighted_cells.append(aligned_element)

    #   if meta["title"] in sentence:
    #     match_count += 1
    #     highlighted_cells.append({"title": meta["title"]})
    # If the matched cells are not important entity
    # Define unimportant entity: Year, single digit
    def is_unimportant_entity(cell):
      if cell["text"].isdigit() and (len(cell["text"]) == 1 or len(cell["text"]) == 4):
        return True
      return False

    if len(highlighted_cells) == 1 and is_unimportant_entity(highlighted_cells[0]):
      highlighted_cells = []
    return highlighted_cells, annotated_sentence


def data_loader_process(input_file, file_filter, job_queue, num_workers):
  if input_file.is_dir():
    files = list(input_file.glob(file_filter))
    print('Working on {}'.format([f.name for f in files]), file=sys.stderr)
    sys.stderr.flush()
  else:
    files = [input_file]
  pbar = tqdm(file=sys.stdout)
  for file in files:
    print(f'parsing {file}', file=sys.stderr)
    sys.stderr.flush()


    for line in file.open():
      job_queue.put(line)
      pbar.update(1)

  pbar.close()

  for i in range(num_workers):
    job_queue.put(None)

def example_writer_process(output_file, example_queue):
  data = example_queue.get()
  with output_file.open('w') as f:
    while data is not None:
      d = json.dumps(data)
      f.write(d + os.linesep)

      data = example_queue.get()

def process():
  parser = ArgumentParser()
  parser.add_argument('--input_file', type=Path, required=True)
  parser.add_argument('--filter', type=str, default='*.jsonl', required=False)
  parser.add_argument("--table_output_file", type=Path)
  parser.add_argument("--sentence_output_file", type=Path)
  parser.add_argument('--worker_num', type=int, default=multiprocessing.cpu_count() - 1, required=False)
  parser.add_argument("--mode", choices=["EXTRACT_TABLE_HTML", "EXTRACT_CONTEXT_TABLE_PAIR"])
  args = parser.parse_args()

  job_queue = multiprocessing.Queue(maxsize=2000)
  sentence_queue, table_queue = None, None
  if args.mode in ["EXTRACT_CONTEXT_TABLE_PAIR"]:
    sentence_queue = multiprocessing.Queue()
  if args.mode in ["EXTRACT_TABLE_HTML"]:
    table_queue = multiprocessing.Queue()
  num_workers = args.worker_num

  loader = multiprocessing.Process(target=data_loader_process, daemon=True,
                                   args=(args.input_file, args.filter, job_queue, num_workers))
  loader.start()

  workers = []
  for i in range(num_workers):
    worker = TableExtractor(args.mode, job_queue, table_queue, sentence_queue, daemon=True)
    worker.start()
    workers.append(worker)

  if args.mode in []:
    sentence_writer = multiprocessing.Process(target=example_writer_process, daemon=True, args=(args.sentence_output_file, sentence_queue))
    sentence_writer.start()
  if args.mode in ["EXTRACT_TABLE_HTML"]:
    table_writer = multiprocessing.Process(target=example_writer_process, daemon=True, args=(args.table_output_file, table_queue))
    table_writer.start()


  for worker in workers:
    worker.join()
  loader.join()

  if args.mode in ["EXTRACT_TABLE_HTML"]:
    table_queue.put(None)
    table_writer.join()
  if args.mode in []:
    sentence_queue.put(None)
    sentence_writer.join()



def debug():
  parser = ArgumentParser()
  parser.add_argument('--input_file', type=Path, required=True)
  parser.add_argument("--output_file", type=Path, required=True)
  args = parser.parse_args()
  fout = args.output_file.open('w')
  extractor = TableExtractor(None, None, None)
  for line in tqdm(args.input_file.open()):
    example = json.loads(line)
    print(example["url"])
    # context_sents = extractor.extract_context(example["url"], example["data"])
    for table in extractor._extract_table(example["url"], example["data"]):
      fout.write(json.dumps(table) + "\n")
      # matched_sentences = extractor.matching(table, context_sents)
      # for matched_sentence in matched_sentences:
      #   fout.write(json.dumps(matched_sentence) + "\n")

def debug_context_table_pair():
  import urllib3
  from skr2.structure.table import Table, TableConfig
  config = TableConfig()
  http = urllib3.PoolManager()
  urllib3.disable_warnings()
  # url = "https://en.wikipedia.org/wiki/2021_German_federal_election"
  url = "https://en.wikipedia.org/wiki/1984_United_States_presidential_election_in_Illinois"
  r = http.request('GET', url)
  if r.status == 200:
    data = r.data.decode('utf-8')
    extractor = TableExtractor("EXTRACT_TABLE_HTML", None, None, None)

    # for idx, table in enumerate(extractor._extract_table(url, data)):
    #   print(table)
    context_sents = extractor.extract_context(url, data)
    # for sent in context_sents:
    #   print(sent)
    for idx, table in enumerate(extractor.extract_table(url, data)):
      t = Table.from_extractor_annotation(table, config=config)
      matched_sentences = extractor.matching(table, context_sents)
      print("Table {}".format(idx))
      print(t)
      for matched_sentence in matched_sentences:
        print(matched_sentence["sentence"]["text"])
        print([cell["text"] for cell in matched_sentence["highlighted_cells"]])


def debug_page():
  import urllib3
  # from malaina.structure.table import Table, TableConfig
  # config = TableConfig()
  http = urllib3.PoolManager()
  urllib3.disable_warnings()
  # url = "https://en.wikipedia.org/wiki/Koinaka"
  # url = "https://en.wikipedia.org/wiki/?curid=15638856"
  # url = "https://en.wikipedia.org/wiki/?curid=901831"
  # url = "https://en.wikipedia.org/wiki/Sota_Fukushi"
  # url = "https://en.wikipedia.org/wiki/1989_in_the_sport_of_athletics"
  # url = "https://en.wikipedia.org/wiki/Paula_Patton"
  # url = "https://en.wikipedia.org/wiki/1984_United_States_presidential_election_in_Illinois"
  # url = "https://en.wikipedia.org/wiki/1982_Illinois_gubernatorial_election"
  # url = "https://en.wikipedia.org/wiki/?curid=10644783"
  # url = "https://en.wikipedia.org/wiki/?curid=39010140"
  # url = "https://en.wikipedia.org/wiki/?curid=19402582"
  # url = "https://en.wikipedia.org/wiki/?curid=1724217"
  # url = "https://en.wikipedia.org/wiki/?curid=39629412"
  # url = "https://zh.wikipedia.org/wiki/中国男子篮球职业联赛"
  url = "https://en.wikipedia.org/wiki/2021_German_federal_election"
  # url = "https://en.wikipedia.org/wiki?curid=63257765"
  # 10741640 Text as title
  r = http.request('GET', url)
  if r.status == 200:
    data = r.data.decode('utf-8')
    extractor = TableExtractor("EXTRACT_TABLE_HTML", None, None, None)
    # interlingual_link = extractor.extract_interlingual_link(url, data)
    # infobox = extractor.extract_infobox(url, data)
    for idx, table in enumerate(extractor._extract_table(url, data)):
      print(table)
      # pass
    # context_sents = extractor.extract_context(url, data)
    # for sent in context_sents:
    #   print(sent)
    # context_sents = ["In Lee Daniels' critically acclaimed drama film Precious (2009), she played Ms. Blu Rain, a teacher at the alternative high school in Harlem, New York, who teaches and mentors disadvantaged students, including the titular character, Claireece Precious Jones (Gabourey Sidibe)."]
    # context_sents = ["Patton's big break came in 2006 when she landed the pivotal female lead role of Claire Kuchever in the science fiction thriller Déjà Vu alongside Denzel Washington."]
    # context_sents = ["On December 20, 1954, he defeated Roy Ankrah in a fourth-round technical knockout in Paris."]
    # context_sents = ["Robert Cohen (born November 15, 1930, in Bône, French Algeria) is a retired French boxer."]
    # context_sents = ['On December 23, 1954, Cohen was stripped of his title by the National Boxing Association for failing to defend it within 90 days against Raul "Raton" Macias.']
    # context_sents = ["On December 11, 1955 Cohen lost in a ten-round technical knockout against French featherweight champion Cherif Hamia before a crowd of 14,000."]
    # context_sents = ['The album debuted at number 98 on the Billboard 200 chart, with first-week sales of 5,400 copies in the United States.']
    raise NotImplementedError()
    for idx, table in enumerate(extractor.extract_table(url, data)):
      t = Table.from_extractor_annotation(table, config=config)
      matched_sentences = extractor.matching(table, context_sents)
      print("Table {}".format(idx))
      print(t)
      for matched_sentence in matched_sentences:
        print(matched_sentence["sentence"]["text"])
        print([cell["text"] for cell in matched_sentence["highlighted_cells"]])


if __name__ == "__main__":
  # debug()
  # debug_page()
  debug_context_table_pair()
  # process()
  # table_process()