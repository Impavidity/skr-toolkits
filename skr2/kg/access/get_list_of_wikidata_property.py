# Get HTML dump from page https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all
# And parse the table

import argparse
from bs4 import BeautifulSoup
import urllib3
http = urllib3.PoolManager()
import json
from skr2.table.html_table import HtmlTable
from skr2.utils import crash_on_ipy
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    page_url = "https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all"
    r = http.request('GET', page_url)
    tables = []
    properties = []
    if r.status == 200:
        page_content = r.data.decode('utf-8')
        soup = BeautifulSoup(page_content, 'html.parser')
        page_title = soup.title.text  # .replace(" - Wikipedia", "")
        # rs = # soup.find_all(class_='wikitable') + soup.find_all(class_="infobox")
        # rs = soup.find_all("table")
        rs = soup.find_all(lambda tag: tag.name=='table' and not tag.find_parent('table'))
        tables = []
        for index, r in enumerate(rs):
            # print(json.dumps({"html": str(r)}))
            table = BeautifulSoup(str(r), 'html.parser')
            table = HtmlTable(table)
            tables.append(table)
        assert len(tables) == 1 # There is only one giant table of predicates
        rows = tables[0].rows
        for row in rows[1:]:
            properties.append(
                {
                    "id": row[0]["text"],
                    "text": row[1]["text"],
                    "description": row[2]["text"],
                    "aliases": row[3]["text"]
                }
            )
        with open(args.output_path, "w") as fout:
            for item in properties:
                fout.write(json.dumps(item) + "\n")