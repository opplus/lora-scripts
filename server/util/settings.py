import os

from server.util.utils import read_toml_file


def get_dataset_process_config():
    dataset_process_config = None
    dataset_process_config_file = os.path.join(os.getcwd(), "server/conf", "dataset_process_config.toml")
    if os.path.exists(dataset_process_config_file):
        dataset_process_config: dict = read_toml_file(dataset_process_config_file)
    return dataset_process_config


def get_dataset_process_config_by_name(name):
    dataset_process_config = get_dataset_process_config()
    if dataset_process_config is not None and name in dataset_process_config:
        return dataset_process_config[name]
    return None
