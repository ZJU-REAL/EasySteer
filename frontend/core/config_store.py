"""Read-only access to the JSON config presets under frontend/configs/."""

import json
import os

CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs'
)


class ConfigStore:
    """Lists and loads the JSON presets of one or more configs/ subdirs.

    Args:
        subdirs: A single subdir name (e.g. "training"), or a dict mapping
            subdir name to the "type" label reported by list() (a None
            label omits the "type" key, matching the historical responses).
        display_names: Optional mapping from config name to friendly name.

    get() only opens files that are actually enumerated in the subdirs, so
    path traversal via the config name is impossible.
    """

    def __init__(self, subdirs, display_names=None):
        if isinstance(subdirs, str):
            subdirs = {subdirs: None}
        self._dirs = {
            subdir: (os.path.join(CONFIGS_DIR, subdir), type_label)
            for subdir, type_label in subdirs.items()
        }
        self._display_names = display_names or {}

    def list(self):
        """List available configs as {name, display_name[, type]} dicts."""
        configs = []
        for dir_path, type_label in self._dirs.values():
            if not os.path.isdir(dir_path):
                continue
            for filename in sorted(os.listdir(dir_path)):
                if not filename.endswith('.json'):
                    continue
                name = filename[:-5]
                entry = {
                    'name': name,
                    'display_name': self._display_names.get(
                        name, name.replace('_', ' ').title()
                    ),
                }
                if type_label is not None:
                    entry['type'] = type_label
                configs.append(entry)
        return configs

    def get(self, name):
        """Load a config by name, or return None if it is not whitelisted.

        The whitelist is the set of .json files actually present in the
        configured subdirs; anything else (including traversal attempts)
        returns None.
        """
        filename = f'{name}.json'
        for dir_path, _ in self._dirs.values():
            if not os.path.isdir(dir_path):
                continue
            if filename in os.listdir(dir_path):
                with open(
                    os.path.join(dir_path, filename), 'r', encoding='utf-8'
                ) as f:
                    return json.load(f)
        return None
