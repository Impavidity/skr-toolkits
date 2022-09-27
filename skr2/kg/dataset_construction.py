import argparse
import os
import json
import random
random.seed(1234)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="web_nlg")
    parser.add_argument("--dataset_name", type=str, default="web_nlg")
    parser.add_argument("--dataset_config_name", type=str, default="release_v3.0_en")
    parser.add_argument("--local_file_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    if args.dataset == "local":
        assert args.local_file_path is not None
        dataset = {} # q -> [{}]
        with open(args.local_file_path) as fin:
            for line in fin:
                # group into example level
                ex = json.loads(line)
                if ex["sentence"] not in dataset:
                    dataset[ex["sentence"]] = []
                dataset[ex["sentence"]].append({
                    "triple": ex["triple"].split("<S>")[1:],
                    "label": ex["label"]
                })
        dataset = list(dataset.items())
        random.shuffle(dataset)
        train_dataset = dataset[:int(len(dataset) * 0.8)]
        dev_dataset = dataset[int(len(dataset) * 0.8):int(len(dataset))]
        os.makedirs(args.output_dir, exist_ok=True)
        for split, name in zip([train_dataset, dev_dataset], ["train", "validation"]):
            with open(os.path.join(args.output_dir, "{}.json".format(name)), "w") as fout:
                for example in split:
                    for anno in example[1]:
                        fout.write(json.dumps({
                            "sentence": example[0],
                            "triple": anno["triple"],
                            "label": anno["label"]
                        }) + "\n")
            

