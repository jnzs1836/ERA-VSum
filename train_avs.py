import argparse
from data.feature_loader import get_feature_loader
from configs import get_config
from runner import Runner
from data import get_feature_loader


def main(args):
    loader = get_feature_loader(args.dataset_path)
    for features, filename in loader:
        pass


if __name__ == '__main__':
    config = get_config(mode='train')
    for i, split in enumerate(config.splits):
        if i not in config.split_ids:
            print("skip {}".format(i))
            continue
        train_loader = get_feature_loader(config.video_path, config.splits[i]['train_keys'], config.with_images,
                                          config.image_dir, config.video_dir,
                                          mapping_file_path=config.mapping_file)
        test_loader = get_feature_loader(config.video_path, config.splits[i]['test_keys'], config.with_images,
                                         config.image_dir, config.video_dir,
                                         mapping_file_path=config.mapping_file)
        runner = Runner(config, train_loader, test_loader, split_id=i)
        runner.build()
        runner.train()
