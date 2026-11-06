import os
from typing import Any

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf


def _cast_to_config(obj: Any) -> Any:
    """
    Convert dictionaries to DictConfig objects.

    Args:
        obj: The object to convert

    Returns:
        DictConfig if obj is a dict, otherwise obj unchanged
    """
    # if given a dict, return DictConfig instead
    if isinstance(obj, dict):
        return DictConfig(obj, flags={"allow_objects": True})
    return obj


class DictLoader:
    """
    Load YAML config files, merging several in order, into plain containers that may
    define lazily-constructed objects (``_target_``).
    """

    @staticmethod
    def load(filenames: str | list[str], keys: None | str | tuple[str, ...] = None) -> Any:
        loaded = DictLoader._load(filenames, keys)
        return OmegaConf.to_container(loaded, resolve=True)  # resolve to basic container

    @staticmethod
    def _load(filenames: str | list[str], keys: None | str | tuple[str, ...] = None) -> Any:
        """
        Load a config file or multiple config files.

        Args:
            filenames: A single file path or a list of file paths to load.
                       If a list is provided, later files in the list will override
                       the configurations from earlier files.
            keys: Keys to load and return. If not given, return all keys
                  (whose values are config objects) in a dict.

        Returns:
            An OmegaConf config object or a specific subset based on keys.
        """
        # Validate input types first
        if not isinstance(filenames, (str, list)):
            raise TypeError("filenames must be a string or a list of strings.")

        has_keys = keys is not None

        def _handle_keys(
            cfg: DictConfig | ListConfig, keys: None | str | tuple[str, ...], has_keys: bool
        ) -> DictConfig | ListConfig | Any | tuple:
            """
            Extract specific keys from a config object.

            Args:
                cfg: The configuration object
                keys: The keys to extract
                has_keys: Whether keys are provided

            Returns:
                Either the full config, a single value, or a tuple of values
            """
            if not has_keys:
                return cfg
            if isinstance(keys, str):
                return _cast_to_config(cfg[keys])  # type: ignore
            assert keys is not None
            return tuple(_cast_to_config(cfg[k]) for k in keys)  # type: ignore

        if isinstance(filenames, str):
            # Single file load behavior
            filename = os.path.expanduser(filenames).replace("/./", "/")  # redundant
            if os.path.splitext(filename)[1] not in [".yaml", ".yml"]:
                raise ValueError(f"Config file {filename} has to be a yaml file.")

            with open(filename) as f:
                obj = yaml.unsafe_load(f)
            #  ret = OmegaConf.create(obj, flags={"allow_objects": True})
            ret = OmegaConf.create(obj)
            return _handle_keys(ret, keys, has_keys)

        elif isinstance(filenames, list):
            # Multiple files load and merge behavior
            if not filenames:
                raise ValueError("No configuration files provided to load.")

            # merged_cfg = OmegaConf.create(flags={"allow_objects": True})
            merged_cfg = OmegaConf.create()

            for filename in filenames:
                cfg = DictLoader.load(filename)
                merged_cfg = OmegaConf.merge(merged_cfg, cfg)

            return _handle_keys(merged_cfg, keys, has_keys)
