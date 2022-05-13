# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors, The Google AI Language Team Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The WikiTableQuestions dataset is for the task of question answering on semi-structured HTML tables"""
# TODO： This code can be push to HuggingFace as a new contribution.
import ast
import csv
import os
import json

import datasets

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
"""

_DESCRIPTION = """\
"""

_HOMEPAGE = ""

_LICENSE = "CC-BY-SA-4.0 License"

_URL = {
    "train": "https://git.uwaterloo.ca/p8shi/malaina/-/raw/master/totto/totto_train_data_alignment_classification.jsonl",
    "dev": "https://git.uwaterloo.ca/p8shi/malaina/-/raw/master/totto/totto_dev_data_alignment_classification.jsonl"
}


def _load_table(table_path) -> dict:
    """
    attention: the table_path must be the .tsv path.
    Load the WikiTableQuestion from csv file. Result in a dict format like:
    {"header": [header1, header2,...], "rows": [[row11, row12, ...], [row21,...]... [...rownm]]}
    """

    def __extract_content(_line: str):
        _vals = [_.replace("\n", " ").strip() for _ in _line.strip("\n").split("\t")]
        return _vals

    with open(table_path, "r") as f:
        lines = f.readlines()

        rows = []
        for i, line in enumerate(lines):
            line = line.strip('\n')
            if i == 0:
                header = line.split("\t")
            else:
                rows.append(__extract_content(line))

    table_item = {"header": header, "rows": rows}

    # Defense assertion
    for i in range(len(rows) - 1):
        if not len(rows[i]) == len(rows[i - 1]):
            raise ValueError('some rows have diff cols.')

    return table_item


class ToTToContextAligning(datasets.GeneratorBasedBuilder):
    """The processed ToTTo dataset for Context Aligning dataset"""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "statement": datasets.Value("string"),
                    "table": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URL)
        # Use local path for now
        # data_dir = "/home/p8shi/relogic-sql/data/examples/tablekit/ToTTo/totto_data/"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_dir["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_dir["dev"]},
            )
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        # data_id, question, table_id, gold_result_str
        with open(filepath, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                ex = json.loads(line)
                yield idx, {
                    "id": str(idx),
                    "table": ex["table"],
                    "statement": ex["text"],
                    "label": ex["label"]
                }
