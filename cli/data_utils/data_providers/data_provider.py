import os
from pathlib import Path

# This module assumes that the following environment variables are set:
# FEATURES_ROOT: path to the root folder containing features csv files
# SAMPLES_IDX_ROOT: path to the root folder containing samples idx csv files, each
# file will contain on a given row: rgb_file_path, depth_file_path, feature_file_path
FEATURES_ROOT = os.getenv("FEATURES_ROOT")
SAMPLES_IDXS_ROOT = os.getenv("SAMPLES_IDX_ROOT")


"""
For a given data provider we assume the following folder structure:

<data_provider_root_path>/
├── <dataset_name>/
│   ├── <subpath_to_rgbs_folder>
│   │  │── rgb_0_file
│   │  │── rgb_1_file
│   │  │── ...
│   ├── <subpath_to_depth_folder>
│   │  │── depth_0_file
│   │  │── depth_1_file
│   │  │── ...
│   └── ...
├── <dataset_name>/
│   │   ├── <subpath_to_rgbs_folder>
│   │   │  │── rgb_0_file
│   │   │  │── rgb_1_file
│   │   │  │── ...
│   │   ├── <subpath_to_depth_folder>
│   │   │  │── depth_0_file
│   │   │  │── depth_1_file
│   │   │  │── ...
│   └── ...
└── ...

The <subpath_to_rgbs_folder> and <subpath_to_depth_folder> are relative to the <dataset_name> folder and
should be the same for all datasets of a given data provider.
"""


class DataProvider:
    def __init__(self, name: str, root_path: str, rgb_root: str, depth_root: str):

        if not FEATURES_ROOT:
            raise ValueError("FEATURES_ROOT environment variable is not set.")
        if not SAMPLES_IDXS_ROOT:
            raise ValueError("SAMPLES_IDX_ROOT environment variable is not set.")
        
        self.name = name
        # The root path of the dataset is the one that contains the depth and rgb folders
        # the providers do not provide features, aditionally we assume that every folder inside the
        # root path is a dataset
        self.root_path = Path(root_path)
        self.rgb_root = Path(rgb_root)
        self.depth_root = Path(depth_root)
        #self.features_root = Path(FEATURES_ROOT)
        #self.samples_idx_root = Path(SAMPLES_IDXS_ROOT)
        
    def list_datasets(self):
        return [f.name for f in Path(self.root_path).iterdir() if f.is_dir()]

    def get_datasets(self):
        return [DepthDataset(self, dataset_name) for dataset_name in self.list_datasets()]