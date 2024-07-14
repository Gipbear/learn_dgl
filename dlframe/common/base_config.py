import yaml
from yaml import SafeLoader, SafeDumper
from yaml.nodes import MappingNode
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Type
from typing_extensions import Self


@dataclass
class BaseConfig(object):
    @classmethod
    def constructor(cls, loader: SafeLoader, node: MappingNode) -> Self:
        """Construct an instance."""
        return cls(**loader.construct_mapping(node))

    @classmethod
    def loader(cls, safe_loader: SafeLoader) -> Type[SafeLoader]:
        """Add constructors to PyYAML loader."""
        safe_loader = yaml.SafeLoader
        safe_loader.add_constructor(f"!{cls.__name__}", cls.constructor)
        for (name, data_fields) in cls.__dataclass_fields__.items():
            cls_type = data_fields.type
            if is_dataclass(cls_type):
                safe_loader.add_constructor(f"!{cls_type.__name__}", cls_type.constructor)
                safe_loader = cls_type.loader(SafeLoader)
        return safe_loader

    @classmethod
    def representer(cls, dumper: SafeDumper, config) -> MappingNode:
        """Represent an instance as a YAML mapping node."""
        return dumper.represent_mapping(f"!{cls.__name__}", config.__dict__)

    @classmethod
    def dumper(cls, safe_dumper: SafeDumper) -> Type[SafeDumper]:
        """Add representers to a YAML seriailizer."""
        safe_dumper.add_representer(cls, cls.representer)
        for (name, data_fields) in cls.__dataclass_fields__.items():
            cls_type = data_fields.type
            if is_dataclass(cls_type):
                safe_dumper.add_representer(cls_type, cls_type.representer)
                safe_dumper = cls_type.dumper(safe_dumper)
        return safe_dumper


@dataclass
class DatasetConfig(BaseConfig):
    name: str = ""
    data_root_path: str = f"./data/{name}"


@dataclass
class ModelHyperConfig(BaseConfig):
    n_input: int = -1
    n_hidden: int = -1
    n_output: int = -1
    ...


@dataclass
class TrainHyperConfig(BaseConfig):
    lr: float = 0.005  # 学习率
    criterion: str = ""  # 损失函数
    optimizer: str = ""  # 优化器
    scheduler: str = ""  # 学习率调整计划


@dataclass
class modelConfig(BaseConfig):
    datasetConfig: DatasetConfig = field(default_factory=DatasetConfig)
    modelHyperConfig: ModelHyperConfig = field(default_factory=ModelHyperConfig)
    trainHyperConfig: TrainHyperConfig = field(default_factory=TrainHyperConfig)


if __name__ == "__main__":
    # 加载 yaml
    # config = modelConfig()
    # config = yaml.load(open("config.yaml", "rb"), Loader=modelConfig.loader(SafeLoader))
    # print(config)
    # print(asdict(config))
    # 保存 yaml
    config = modelConfig()
    print(config)
    print(asdict(config))
    with open("config.yaml", 'w') as f:
        yaml.dump(config, f, Dumper=modelConfig.dumper(SafeDumper), sort_keys=False)
