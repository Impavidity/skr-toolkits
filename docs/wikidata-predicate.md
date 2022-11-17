# Dump Wikidata Properties

We dump all the wikidata properties from [this link](https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all),
by parsing the giant table and saving the properties in `jsonl` format with `id`, `label`, `description` and `aliases`.
An example is shown here.
```json
{"id": "P35", "text": "head of state", "description": "official with the highest formal authority in a country/state", "aliases": "leader, president, queen, monarch, king, emperor, governor, state headed by, chief of state"}
```

To obtain the dump, run the following code:
```bash
python -m skr2.kg.access.get_list_of_wikidata_property --output_path wikidata_properties.json
```

The `jsonl` file will be saved in `wikidata_properties.json`.