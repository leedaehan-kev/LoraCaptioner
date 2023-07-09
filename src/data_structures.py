import yaml


class AttrDict(dict):
    """Convert a python built-in dict to an object with key-values as attributes.
    This is mainly to maintain compatibility with wandb config, which makes values as attributes.
    Example:
        d = AttrDict({'key': 'value'})
        print(d.key)  # Outputs: 'value'
    """
    def __init__(self, dict: dict):
        super(AttrDict, self).__init__(dict)
        self.__dict__ = self


if __name__ == '__main__':
    # Test 1: Read yaml file
    with open("../configs/default.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrDict(config)
    print(config.batch_size_per_gpu)
    print(config.lora_config)
