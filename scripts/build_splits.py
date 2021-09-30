import h5py
import sys
import argparse
import json
import random


sys.path.append("../")

parser = argparse.ArgumentParser()
parser.add_argument('--h5_path', type=str, default='/your/h5/file/path')
parser.add_argument("--test_video", type=str, default="/your_video_name")
parser.add_argument("--output_path", type=str, default="/your/output/path/tvsum_splits.json")
parser.add_argument("--non_overlap", type=bool, default=True)
_args = parser.parse_args()


def build_non_overlap_splits(all_keys, split_num = 5):
    shuffled_keys = random.shuffle(all_keys)
    shuffled_keys = all_keys
    print(shuffled_keys)
    test_split_len = len(all_keys) // split_num
    splits = []
    for i in range(split_num):
        test_keys = shuffled_keys[i * test_split_len: (i + 1)*test_split_len]
        train_keys = list(filter(lambda x: x not in test_keys, shuffled_keys))
        print(test_keys)
        split = {
            "train_keys": train_keys,
            "test_keys": test_keys
        }
        splits.append(split)
    return splits


def build_urban_splits(all_keys, test_video):
    train_keys = []
    test_keys = []
    for video_split_name in all_keys:
        video_name = "_".join(video_split_name.split("_")[:-2])
        if video_name == test_video:
            test_keys.append(video_split_name)
        else:
            train_keys.append(video_split_name)
    splits_info = {
        "train_keys": train_keys,
        "test_keys": test_keys
    }
    return [splits_info]


def main(args):
    hf = h5py.File(args.h5_path, 'r')
    all_keys = list(hf.keys())
    if args.non_overlap:
        splits = build_non_overlap_splits(all_keys)
    else:
        splits = build_urban_splits(all_keys, args.test_video)

    with open(args.output_path, "w") as fp:
        json.dump(splits, fp)


if __name__ == '__main__':
    main(_args)
