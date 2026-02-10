#! /usr/bin/env python3

"""train.py: Trains a new model from config file."""

import utils
with utils.DiscoverSourcePath():
    import Framework
    from Implementations import Methods as MI
    from Implementations import Datasets as DI


def main(config_path: str = None):
    Framework.setup(config_path=config_path, require_custom_config=True)
    training_instance = MI.get_training_instance(
        method=Framework.config.GLOBAL.METHOD_TYPE,
        checkpoint=Framework.config.TRAINING.LOAD_CHECKPOINT
    )
    dataset = DI.get_dataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH
    )
    training_instance.run(dataset)
    Framework.teardown()
    return training_instance


if __name__ == '__main__':
    main()
