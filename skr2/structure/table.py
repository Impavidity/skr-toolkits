import dataclasses
import hashlib
import random
from typing import List, Optional
import pandas as pd
from tabulate import tabulate
import enum
from transformers.models.tapas.tokenization_tapas import parse_text
from skr2.structure.template import _to_floats, convert_to_float
import os
import json
import sqlite3
import spacy

nlp = spacy.load('en_core_web_sm')

_MAX_INT = 2**32 - 1

def fingerprint(text):
  return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)

def clean_text(text):
  if len(text.strip()) == 0:
    return "none"
  return " ".join(text.split())

def clean_token_for_column_name(token):
  for st in """. + - ? ! * @ % ^ & # = / \ : " ' ( )""".split():
    token = token.replace(st, "")
  return token

def convert_text_to_column_name(text):
  tokens = text.split()
  tokens = [clean_token_for_column_name(token) for token in tokens]
  return "_".join(tokens).lower()


def _create_rank_dict(numbers):
  return {f: rank for rank, f in enumerate(sorted(set(numbers)))}


def table_linearization(table_array):
  strs = ["col: {}".format(" | ".join(table_array[0]))]
  for row_idx, row in enumerate(table_array[1:]):
    strs.append("row {}: {}".format(row_idx, " | ".join(row)))
  return " | ".join(strs)

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

def annotate_cell_values(rows, nlp):
  annotations = []
  schema = {}
  if len(rows) > 0:
    for col_idx, cell_value in enumerate(rows[0]):
      schema[col_idx] = cell_value
  for row_idx, row in enumerate(rows):
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
    annotations.append(annotation)
  return annotations

class SamplingTask(enum.Enum):
  RETRIEVAL = 0

@dataclasses.dataclass
class QueryTargetPair:
  query: str = None
  target: str = None

@dataclasses.dataclass(frozen=True)
class TableConfig:
  """Table configuration

    min_rows
  """
  min_row_num: int = 4
  max_row_num: int = 30
  min_col_num: int = 2
  ignore_hierarchical_schema_table: bool = True
  ignore_empty_column_table: bool = True
  infer_column_type: bool = True
  retrieval_sampling_right_boundary: int = None
  retrieval_relevant_cell_length_limit: int = None
  sampling_limit: int = 10
  prob_count_aggregation: float = 0.2
  prob_stop_value_row_expansion: float = 0.5
  sql_populate_size: int = 3
  no_process: bool = False


@dataclasses.dataclass
class Date:
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None


@dataclasses.dataclass
class NumericValue:
    float_value: Optional[float] = None
    date: Optional[Date] = None


@dataclasses.dataclass
class NumericValueSpan:
    begin_index: int = None
    end_index: int = None
    values: List[NumericValue] = None


class Column:
  def __init__(self,
               column_id=None,
               name=None,
               text=None,
               column_type=None,
               is_primary_key=None,
               foreign_key=None,
               table=None,
               cell_coordinate=None):
    self.column_id = column_id
    self.name = name
    self.text = text
    self.column_type = column_type
    self.is_primary_key = is_primary_key
    self.foreign_key = foreign_key
    self.table = table
    self.cell_coordinate = cell_coordinate
  
  @property
  def row_index(self):
    return 0
  
  @property
  def column_index(self):
    return self.column_id

  def __len__(self):
    return len(self.name)

  def is_valid(self, mode=None):
    if mode == SamplingTask.RETRIEVAL:
      return True

  def __repr__(self):
    return json.dumps(self.__dict__)

class Row:
  def __init__(self, row_id, row):
    self.row_id = row_id
    self.row = row

  def __getitem__(self, item):
    if item < len(self.row):
      return self.row[item]
    else:
      return Cell(text="none", column=None, table=None, row_index=None, column_index=None)

  def linearize(self):
    row = []
    for cell in self.row:
      if cell.text:
        row.append("{} is {}".format(cell.column.name, cell.text))
    return ", ".join(row) + "."

  def __len__(self):
    return len(self.row)

  def __iter__(self):
    for item in self.row:
      yield item


class Cell:
  def __init__(self, text, column, table, row_index, column_index):
    self.text = text
    self.alias = [text]
    self.column = column
    self.table = table
    self.numeric_value = parse_text(text)
    self.row_index, self.column_index = row_index, column_index
    self.rank = None

  @property
  def random_alias(self):
    return self.table.rng.choice(self.alias)

  def __len__(self):
    return len(self.text.split())

  def __eq__(self, other):
    return self.text == other.text

class SQLTemplate:
  def __init__(self, template):
    template_tokens = []
    for token in template.split():
      if token == "ID":
        template_tokens.append("ROWID")
      else:
        template_tokens.append(token)
    self.template = " ".join(template_tokens)

  def populate_one_sql(self, table):
    sql_tokens = []
    unfill_column = None
    column_dict = {}
    for token in self.template.split():
      if token.startswith("[") and token.endswith("]"):
        # We need to fill in the token
        if token.startswith("[COLUMN"):
          # need to fill in the column
          if token in column_dict:
            value = column_dict[token].name
            unfill_column = column_dict[token]
          else:
            column_type = token.split("_")[1]
            column = table.sample_column(column_type, sampled_columns=column_dict.values())
            if column is None:
              return None
            column_dict[token] = column
            unfill_column = column
            value = column.name
        elif token == "[LIMIT_VALUE]":
          value = str(1)
        else:
          if unfill_column is not None:
            if unfill_column.column_type != token[1:-1]:
              if token != "[NUMBER]":
                # print(self.template)
                return None
              else:
                if sql_tokens[-2] != ")":
                  # print(self.template)
                  return None
                else:
                  value = str(2)
            else:
              value = table.sample_value(unfill_column)
              if not value:
                return None
              if unfill_column.column_type == "TEXT":
                value = json.dumps(value)
              else:
                value = json.dumps(convert_to_float(value))
          else:
            # print(self.template)
            return None
            # value = str(2)
        sql_tokens.append(value)
      else:
        sql_tokens.append(token)
    return " ".join(sql_tokens)


class Table:
  def __init__(self,
               config=None,
               table=None,
               page_title=None,
               section_titles=None,
               table_caption=None,
               page_url=None,
               page_table_index=None,
               annotations=None,
               table_id=None):
    self.config = config
    self.table = table
    self.page_title = page_title
    self.section_titles = section_titles
    self.table_caption = table_caption
    self.page_url = page_url
    self.page_table_index = page_table_index
    self.annotations = annotations

    if not self.config.no_process:
      """
      We assume the first row is the column for now
      TODO: Improve the schema and contents based on the structure of the table.
      """
      self.columns = self._construct_columns(self.table[0])
      self.contents = self._construct_contents(self.table[1:])
      self.create_rank()
      self.type_to_columns = self.classify_column_types()

    self.data_frame = None

    self.iid = table_id if table_id else "{}-{}".format(self.page_url.split("=")[-1], self.page_table_index)
    self.rng = random.Random(fingerprint(repr(self.iid)) % _MAX_INT)

  def get_cell(self, row_index, column_index):
    if self.config.no_process:
      return Cell(
        text=self.table[row_index][column_index],
        table=self,
        column=None,
        row_index=row_index,
        column_index=column_index)
    else:
      if row_index == 0:
        return self.columns[column_index]
      else:
        return self.contents[row_index-1][column_index]

  def linearize_table(self, use_column_name=False):
    if self.config.no_process:
      return self.table
    else:
      if use_column_name:
        columns = [column.name for column in self.columns]
      else:
        columns = [column.text for column in self.columns]
      contents = [[cell.text for cell in row] for row in self.contents]
      return [columns] + contents

  def classify_column_types(self):
    type_to_column = {"TEXT": [], "NUMBER": [], "GROUPED_TEXT": []}
    if len(self.contents) > 0:
      for idx, cell in enumerate(self.contents[0]):
        if cell.rank is not None:
          self.columns[idx].column_type = "NUMBER"
        else:
          self.columns[idx].column_type = "TEXT"
        type_to_column[self.columns[idx].column_type].append(self.columns[idx])
        if self.columns[idx].column_type == "TEXT":
          # Check if it is group text
          cell_values = set()
          for row in self.contents:
            if len(row[idx].text) > 0:
              if row[idx].text in cell_values:
                type_to_column["GROUPED_TEXT"].append(self.columns[idx])
                break
              else:
                cell_values.add(row[idx].text)
    return type_to_column

  def create_rank(self):
    for column_index in range(len(self.columns)):
      try:
        values = [row[column_index].text for row in self.contents]
        floats = _to_floats(values)
        rank_dict = _create_rank_dict(floats)
        ranks = [rank_dict[f] for f in floats]
        if len(ranks) != len(self.contents):
          raise ValueError()
        for row_index in range(len(ranks)):
          self.contents[row_index][column_index].rank = ranks[row_index]
      except:
        continue

  def _construct_columns(self, columns):
    _columns = []
    for column_idx, column in enumerate(columns):
      _columns.append(Column(column_id=column_idx,
                            name=convert_text_to_column_name(column),
                            text=clean_text(column),
                            column_type=None))
    return _columns

  def _construct_contents(self, contents):
    _contents = []
    for row_index, row in enumerate(contents):
      _content = []
      for cell, column in zip(row, self.columns):
        _content.append(Cell(text=clean_text(cell), column=column,
                             table=self,
                             column_index=column.column_id,
                             row_index=row_index))
      _contents.append(Row(row_id=row_index + 1, row=_content))
    return _contents


  def is_valid(self, column_vocabs=None):
    if len(self.table) < self.config.min_row_num:
      return False
    if len(self.table) > self.config.max_row_num:
      # Ignore large tables
      return False

    if self.config.ignore_hierarchical_schema_table:
      column_name_set = set()
      for column in self.columns:
        if len(column) > 0: # not empty string for the column
          if column.name not in column_name_set:
            column_name_set.add(column.name)
          else:
            return False
        else:
          if self.config.ignore_empty_column_table:
            return False

    if column_vocabs is not None:
      tolerance = 0
      for column in self.columns:
        if column.text not in column_vocabs:
          tolerance += 1
      if tolerance > 2:
        return False

    return True

  def __repr__(self):
    return tabulate(self.table, tablefmt="grid")

  def dataframe(self):
    if self.data_frame is None:
      self.data_frame = pd.DataFrame(data=[[cell.text for cell in row] for row in self.contents], columns=[column.text for column in self.columns])
    return self.data_frame

  def render(self):
    return self.dataframe().style.hide_index()

  def generate_aggregation_query(self, agg_type):
    pass

  def _generate_aggregation_query(self, agg_type):
    pass

  def generate_lookup_query(self):
    pass

  def _generate_lookup_query(self):
    pass

  def _generate_retrieval_query(self):
    # select a column
    column_candidates = []
    for column in self.columns:
      if column.is_valid(mode=SamplingTask.RETRIEVAL):
        column_candidates.append(column)
    column: Column = self.rng.choice(column_candidates[:self.config.retrieval_sampling_right_boundary])


    values = []
    for row in self.contents:
      if column.column_id < len(row) and len(row[column.column_id]) > 0:
        if self.config.retrieval_relevant_cell_length_limit is not None:
          if len(row[column.column_id]) < self.config.retrieval_relevant_cell_length_limit:
            values.append(row[column.column_id])
        else:
          values.append(row[column.column_id])
    if len(values) == 0:
      return None
    value = self.rng.choice(values)

    query = "{} is {}".format(column.text, value.text)
    answer_rows = []
    for row in self.contents:
      if row[column.column_id] == value:
        answer_rows.append(row)

    target = self._linearize_rows(answer_rows)

    return QueryTargetPair(query=query, target=target)

  def _linearize_rows(self, answer_rows):
    rows = []
    for row in answer_rows:
      rows.append(row.linearize())
    return " ".join(rows)

  def generate_retrieval_query(self):
    cnt = 0
    while cnt < self.config.sampling_limit:
      output = self._generate_retrieval_query()
      if output is None:
        cnt += 1
      else:
        return output

  def schema_recovery(self):
    pass

  def row_permutation(self):
    pass

  def value_inferring(self):
    pass

  def populate_sqls(self, templates):
    def linearize_output(output):
      answers = []
      for row in output:
        for item in row:
          answers.append(str(item))
      answer_str = " | ".join(answers)
      if len(answer_str.strip()) == 0 or answer_str == "0" or answer_str == "None":
        return None
      if len(answer_str.split()) > 200:
        return None
      return answer_str

    self.rng.shuffle(templates)
    sqls = []
    for sampled_template in templates:
      if not isinstance(sampled_template, SQLTemplate):
        sampled_template = SQLTemplate(sampled_template)
      sql = sampled_template.populate_one_sql(self)
      if sql:
        # execute
        output = self.execute_sql(sql)
        linearized_output = linearize_output(output)
        if linearized_output:
          sqls.append({
            "page_url": self.page_url,
            "page_table_index": self.page_table_index,
            "sql": sql,
            "template": sampled_template.template,
            "exec": linearized_output})
      if len(sqls) >= self.config.sql_populate_size:
        break
    return sqls

  def sample_column(self, column_type=None, sampled_columns=list([]), k=1):
    sampled_column_names = [column.name for column in sampled_columns]
    if column_type is not None:
      candidates = []
      for column in self.type_to_columns[column_type]:
        if column.name not in sampled_column_names:
          candidates.append(column)
      if len(candidates) < k:
        return None
      return self.rng.sample(candidates, k=k)
    else:
      # random sample
      candidates = []
      for column in self.columns:
        if column.name not in sampled_column_names:
          candidates.append(column)
      if len(candidates) < k:
        return None
      return self.rng.sample(candidates, k=k)

  def sample_value(self, column):
    column_idx = column.column_id
    row_id = self.rng.randint(0, len(self.contents)-1)
    return self.contents[row_id][column_idx].text

  def create_db(self, path="/home/ec2-user/databases"):
    os.makedirs(path, exist_ok=True)
    db_path = '{}/{}.sqlite'.format(path, self.iid)
    if os.path.exists(db_path):
      con = sqlite3.connect(db_path)
      cur = con.cursor()
      return cur
    else:
      con = sqlite3.connect(db_path)
      cur = con.cursor()
      comd_str = "CREATE TABLE W"
      comd_str += " ( {} )".format(", ".join(["{} {}".format(column.name, column.column_type.lower()) for column in self.columns]))
      cur.execute(comd_str)

      # insert rows
      rows = []
      for row in self.contents:
        rows.append(tuple([item.text for item in row]))

      cur.executemany('insert into W values ({})'.format(",".join(["?"] * len(rows[0]))), rows)
      con.commit()
      return cur


  def execute_sql(self, sql):
    output = []
    try:
      cur = self.create_db()
    except:
      return output
    try:
      for item in cur.execute(sql):
        output.append(item)
    except:
      pass
    return output


  @classmethod
  def from_extractor_annotation(cls, d, config):
    """
    Construct class from the output of HTML table extractor.
    :param d: json object
    :return: Table class
    """
    try:
      table_array = [[cell["text"] for cell in row] for row in d["annotations"]]
      page_title = d["page_title"]["text"]
      annotations = d["annotations"]
    except:
      table_array = d["annotations"]
      page_title = d["page_title"]
      annotations = annotate_cell_values(table_array, nlp)
    if len(table_array) == 0:
      # raise ValueError("The table is empty")
      return None
    if len(table_array) == 1:
      # raise ValueError("The table only has one row {}".format(" | ".join(table_array[0])))
      return None
    return cls(
      config=config,
      table=table_array,
      page_title=page_title,
      section_titles=d["section_titles"],
      table_caption=d["table_caption"],
      page_url=d["url"],
      annotations=annotations,
      page_table_index=d["idx"])

  @classmethod
  def from_tapas_corpus(cls, d, config):
    """
    {"columns": [{"text": ""}, {"text": ""}, {"text": "Born"}, {"text": "Residence"}, {"text": "Occupation"},
                 {"text": "Years\u00a0active"}, {"text": "Height"}, {"text": "Television"}, {"text": "Children"}],
     "rows": [
     {"cells": [{"text": "Lesley Joseph"}, {"text": "Joseph in Cardiff, Wales, May 2011"},
      {"text": "Lesley Diana Joseph  14 October 1945 (age\u00a072) Finsbury Park, Haringey, London, England"},
      {"text": "Hampstead, North London"}, {"text": "Broadcaster, actress"}, {"text": "1969\u2013present"},
      {"text": "5\u00a0ft 2\u00a0in (1.57\u00a0m)"}, {"text": "Birds of a Feather"}, {"text": "2"}]
     }],
     "tableId": "Lesley Joseph_A1D55A57012E3362",
     "documentTitle": "Lesley Joseph",
     "documentUrl": "https://en.wikipedia.org//w/index.php?title=Lesley_Joseph&amp;oldid=843506707"}
    :param d:
    :param config:
    :return:
    """
    columns = [c["text"] for c in d["columns"]]
    contents = [[c["text"] for c in r["cells"]] for r in d["rows"]]
    return cls(
      config=config,
      table=[columns] + contents,
      page_title=d["documentTitle"],
      page_url=d["documentUrl"],
      table_id=d["tableId"],
    )

  @classmethod
  def from_totto_annotation(cls, d, config):
    columns = [c["value"] for c in d["table"][0]]
    contents = [[c["value"] for c in r] for r in d["table"][1:]]
    if config.no_process:
      annotations = None
    else:
      annotations = annotate_cell_values([columns] + contents, nlp)
    return cls(
      config=config,
      table=[columns] + contents,
      page_title=d["table_page_title"],
      section_titles=d["table_section_title"],
      page_url=d["table_webpage_url"],
      annotations=annotations
    )

from skr2.structure.template import (
  SynthesizationError,
  _synthesize_count_condition,
  _synthesize_aggregation_expression,
)

def synthesize_from_table(table: Table):
  for _ in range(table.config.sampling_limit):
    try:
      if table.rng.random() < table.config.prob_count_aggregation:
        # Create a count example
        return _synthesize_count_condition(table)
      else:
        return _synthesize_aggregation_expression(table)
    except SynthesizationError:
      continue
  raise SynthesizationError("Couldn't synthesize condition")

if __name__ == "__main__":
  from malaina.utils import crash_on_ipy
  # from skr2.structure.table import TableConfig, Table
  import json

  tables = []
  config = TableConfig()
  valid_tables = []
  invalid_tables = []
  construction_error = 0
  with open("/mnt/efs/fs1/relogic-sql-2021/data/examples/wikipedia/en/extraction/4/table.jsonl_") as fin:
    for idx, line in enumerate(fin):
      d = json.loads(line)
      table = Table.from_extractor_annotation(d, config)
      try:
        table = Table.from_extractor_annotation(d, config)
      except:
        construction_error += 1
        continue
      if table is None:
        continue
      if table.is_valid():
        valid_tables.append(table)
        break
      else:
        invalid_tables.append(table)
  template_str = 'SELECT [COLUMN_TEXT_0] FROM W WHERE [COLUMN_TEXT_0] = [TEXT]'
  template_str = 'SELECT ( SELECT [COLUMN_NUMBER_0] FROM W WHERE [COLUMN_TEXT_0] = [TEXT] ) - ( SELECT [COLUMN_NUMBER_0] FROM W WHERE [COLUMN_TEXT_0] = [TEXT] )'
  template = SQLTemplate(template_str)
  table = valid_tables[0]
  print(table)

  sql = template.populate_one_sql(table)
  print(sql)
  output = table.execute_sql(sql)
  print(output)


  # queries = []
  # for i in range(20):
  #   question_answer_pair = synthesize_from_table(valid_tables[0])
  #   print(question_answer_pair.question())
  #   print(question_answer_pair.answer)
  #   queries.append(question_answer_pair)

  raise NotImplementedError()
