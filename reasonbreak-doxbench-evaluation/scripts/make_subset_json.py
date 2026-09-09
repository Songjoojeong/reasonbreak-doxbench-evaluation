import argparse
from geoprivacy_eval.dataset import make_subset_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/json/doxbench.json")
    parser.add_argument("--output", default="data/json/doxbench_20.json")
    parser.add_argument("--size", type=int, default=20)
    args = parser.parse_args()
    print(make_subset_json(args.input, args.output, args.size))
