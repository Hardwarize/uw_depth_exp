import os
from pathlib import Path
import re

import pandas as pd

from cli.cli_extract_feature_points import create_features_file



class DepthDataset:
    def __init__(self, provider_name: str, provider_root_path: str, dataset_name: str, rgb_subpath: str, depth_subpath: str):
        try:
            self.provider_name = provider_name
            self.dataset_name = dataset_name

            # Build the rgb and depth folders paths
            self.rgb_root = Path(provider_root_path) / dataset_name / rgb_subpath
            self.depth_root = Path(provider_root_path) / dataset_name / depth_subpath

            self.samples_file_path = Path(os.getenv("SAMPLES_IDX_ROOT")) / self.dataset_name / "samples.csv"
            self.features_folder_root = Path(os.getenv("FEATURES_ROOT")) / self.dataset_name


            #self.rgb_root = provider.root_path / dataset_name / provider.rgb_root 
            #self.depth_root = provider.root_path / dataset_name / provider.depth_root
            #self.features_root = provider.features_root / dataset_name
            #self.samples_idx = provider.samples_idx_root / (dataset_name + ".csv")
            
            #print([f.name for f in Path(self.rgb_root).iterdir() if f.is_file()])
        except FileNotFoundError:
            print(f"{dataset_name} not found, skipping...")
    

    def describe_dataset(self):
        print(f"Data Provider: {self.provider_name}")
        print(f"Dataset Name: {self.dataset_name}")

        print(f"RGBs folder: {self.rgb_root}")
        print(f"* {len([f for f in self.rgb_root.iterdir() if f.is_file()])} RGB files")
        print(f"Depth folder: {self.depth_root}")
        print(f"* {len([f for f in self.depth_root.iterdir() if f.is_file()])} depth files")
        print()
    

    def create_samples_idx_csv(self):
        id_pattern = r'\d{4,20}'
        rgb_images = [str(f) for f in Path(self.rgb_root).iterdir() if f.is_file()]
        depth_images = [str(f) for f in Path(self.depth_root).iterdir() if f.is_file()]

        rgb_dict = pd.DataFrame([{'sample_id': re.search(id_pattern, x).group(), 'path_rgb': x} for x in rgb_images])
        depth_dict = pd.DataFrame([{'sample_id': re.search(id_pattern, x).group(), 'path_depth': x} for x in depth_images])

        df = pd.merge(rgb_dict, depth_dict, on='sample_id').sort_values(by='sample_id').reset_index(drop=True)
        df['path_features'] = df['sample_id'].apply(lambda x: self.features_folder_root / f"{re.search(id_pattern, x).group()}.csv")
        
        self.samples_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.samples_file_path, index=False, header=False)
    

    def create_features(self):
        create_features_file(self.samples_file_path)

