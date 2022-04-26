import unicodedata

def get_table_caption(table):
  if table.parent.name == "td":
    # It is subtable, try to find caption
    for item in table.previous_siblings:
      if item.name == "b": # bold text as caption
        return item.text

def clean_cell_value(cell_val):
  val = unicodedata.normalize('NFKD', cell_val)
  # val = val.encode('ascii', errors='ignore')
  # val = str(val, encoding='ascii')
  return val

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

def get_cell_spans(cell):
  # Handle invalid rowspan value
  # Peng: Not sure the case but learn this from existing code
  # https://github.com/wenhuchen/OTT-QA/blob/master/table_crawling/pipeline.py#L94
  # https://github.com/google-research/tapas/blob/master/tapas/scripts/preprocess_nq_utils.py#L73
  row_span = cell.get("rowspan", 1)
  try:
    row_span = int(row_span.strip(";"))
  except:
    row_span = 1
  col_span = cell.get("colspan", 1)
  try:
    col_span = int(col_span.strip(";"))
  except:
    col_span = 1
  return row_span, col_span

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
            if link and link.has_attr("title"):
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

class HtmlTable:
  def __init__(self, table):
    self.table = table
    # self.page_title = None
    # self.caption = None if table.caption is None else table.caption.text.strip()
    # if self.caption is None:
    #   self.caption = get_table_caption(table)
    # if self.caption:
    #   pass
    self.remove_hidden()

    # self.section_titles = get_section_titles(self.table)
    self.get_cells()
    attrs = next(iter(table)).attrs
    self.is_table_infobox = "infobox" in attrs["class"] \
      if "class" in attrs else False

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

  def get_cells(self):
    # Get all cells
    self.rows = []
    for x in self.table.find_all('tr'):
      row = []
      for y in x.find_all(['th', 'td']):
        # print(y)
        try:
          cell_text = get_cell_text(y)
        except:
          cell_text = ""
          print("Can not parse cell.")
          print(str(y))
        row_span, col_span = get_cell_spans(y)

        # print(cell_text, row_span, col_span)
        row.append({"text": cell_text,
                   "rowspan": row_span,
                   "colspan": col_span,
                   "is_header": y.name == "th"})
      self.rows.append(row)


if __name__ == "__main__":
  from bs4 import BeautifulSoup
  import urllib3
  http = urllib3.PoolManager()
  import json
  # page_url = "https://zh.wikipedia.org/wiki/%E6%9D%B1%E6%80%A5%E9%9B%BB%E9%90%B5"
  # page_url = "https://en.wikipedia.org/wiki/?curid=19202639"
  # page_url = "https://en.wikipedia.org/wiki/?curid=10644783"
  # page_url = "https://en.wikipedia.org/wiki?curid=5094710"
  # page_url = "https://en.wikipedia.org/w/index.php?title=90th_Academy_Awards&oldid=773920751"
  page_url = "https://en.wikipedia.org//w/index.php?title=East_India_Company&amp;oldid=850568847"
  r = http.request('GET', page_url)
  tables = []
  if r.status == 200:
    page_content = r.data.decode('utf-8')
    soup = BeautifulSoup(page_content, 'html.parser')
    page_title = soup.title.text  # .replace(" - Wikipedia", "")
    # rs = # soup.find_all(class_='wikitable') + soup.find_all(class_="infobox")
    # rs = soup.find_all("table")
    rs = soup.find_all(lambda tag: tag.name=='table' and not tag.find_parent('table'))
    for index, r in enumerate(rs):
      # print(json.dumps({"html": str(r)}))
      table = BeautifulSoup(str(r), 'html.parser')
      table = HtmlTable(table)
      print(table.rows)
      print(table.is_table_infobox)
    #   print(table.rows)
  # import json
  # with open("data/examples/wikipedia/en/tables/seg_0.jsonl") as fin:
  #   for line in fin:
  #     example = json.loads(line)
  #     table_html = example["html"].replace("<br />", "<br /> ")
  #     table = BeautifulSoup(table_html, "html5lib").find("table")
  #     html_table = HtmlTable(table)
  #     raise NotImplementedError()

