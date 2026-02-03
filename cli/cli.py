import csv
import random

import typer
from typing import Optional
from typing_extensions import Annotated

import cv2
import pandas as pd

from cli.data_utils.data_providers import DataProviders, DataProvider
from cli.data_utils.depth_dataset import DepthDataset
from cli.cli_train import train_UDFNet

app = typer.Typer()


def build_dataset(dp: DataProvider, dataset_name: str):
    return DepthDataset(
                        provider_name = dp.name,
                        provider_root_path = dp.root_path,
                        dataset_name = dataset_name,
                        rgb_subpath = dp.rgb_root,
                        depth_subpath = dp.depth_root
                    )


# This function is required by typer
@app.callback()
def callback():
    """
    CLI for managing data providers.
    """


@app.command()
def list_dataproviders():
    print("Available Data Providers:")
    for data_provider in DataProviders:
        print(f" - {data_provider.name}")


@app.command()
def check_dataprovider(name: str):
    print(f"Checking data provider: {name}")
    choosen_dp = [dp for dp in DataProviders if dp.name == name]
    if len(choosen_dp) == 0:
        print(f"Data provider {name} not found")
    else:
        print(f"Data provider {name} found")


@app.command()
def list_datasets(name: Annotated[Optional[str], typer.Argument()] = None):
    
    if name is None:
        print("Listing datasets for all providers:")
        for dp in DataProviders:
            print(f"Provider: {dp.name}")
            try:
                datasets = dp.list_datasets()
                for dataset in datasets:
                    print(f" - {dataset}")
            except Exception as e:
                print(f" - Error listing datasets: {e}")
        return


@app.command()
def check_datasets(name: Annotated[Optional[str], typer.Argument()] = None):
    
    if name is None:
        print("Listing datasets for all providers:")
        for dp in DataProviders:
            print(f"Provider: {dp.name}")
            try:
                datasets = dp.list_datasets()
                for dataset in datasets:
                    ds = build_dataset(dp, dataset)
                    ds.describe_dataset()
            except Exception as e:
                print(f" - Error listing datasets: {e}")
        return


@app.command()
def create_samples_idx_file(name: Annotated[Optional[str], typer.Argument()] = None):
    
    if name is None:
        print("Listing datasets for all providers:")
        for dp in DataProviders:
            print(f"Provider: {dp.name}")
            try:
                datasets = dp.list_datasets()
                for dataset in datasets:
                    ds = build_dataset(dp, dataset)
                    ds.create_samples_idx_csv()
            except Exception as e:
                print(f" - Error listing datasets: {e}")
        return


@app.command()
def create_features(name: Annotated[Optional[str], typer.Argument()] = None):
    
    for dp in DataProviders:
        if name in [None, dp.name]:
            print(f"Provider: {dp.name}")
            try:
                datasets = dp.list_datasets()
                for dataset in datasets:

                    ds = build_dataset(dp, dataset)
                    ds.create_features()
            except Exception as e:
                print(f" - Error listing datasets: {e}")
    return


@app.command()
def check_features_random_image():
    dp = random.choice(DataProviders)
    dataset_name = random.choice(dp.list_datasets())
    ds = build_dataset(dp, dataset_name)
    line = random.choice([line for line in csv.reader(open(ds.samples_file_path).read().splitlines())])
    rgb_path = line[1]
    features_path = line[3]
    print(f"Generating feature points image for: {rgb_path}")
    print(f"Feature points stored in: {features_path}")

    img = cv2.resize(cv2.imread(rgb_path), (320, 240))
    df = pd.read_csv(features_path)
    for _, r in df.iterrows(): cv2.circle(img, (int(r['column']), int(r['row'])), 3, (0, 255, 0), -1)

    # Resultado
    cv2.imwrite('kp_results.jpg', img)
    

@app.command()
def train(
    train_ds: Annotated[str, typer.Option("-train", help="String for train datasets")],
    val_ds: Annotated[str, typer.Option("-val", help="String for val datasets")],
    pretrained_model: Annotated[Optional[str], typer.Option("-pretrained", help="Path to pretrained model")] = None
):

    def get_csvs(split_str):
        samples_csvs = []

        data_provider_groups = split_str.split('|')
        for dp_group in data_provider_groups:
            dp_name, ds_str = dp_group.split(':')
            ds_names = ds_str.split(',')

            dp = [dp for dp in DataProviders if dp.name==dp_name][0]
            ds_names = list(filter(lambda x: x in ds_names, dp.list_datasets()))

            for dataset in ds_names:
                ds = build_dataset(dp, dataset)

                samples_csvs.append(ds.samples_file_path)
        
        return samples_csvs
    
    train_samples_csvs = get_csvs(train_ds)
    val_samples_csvs = get_csvs(val_ds)

    train_UDFNet(train_samples = train_samples_csvs, val_samples = val_samples_csvs, pretrained_model=pretrained_model)


@app.command()
def inference():
    from cli.cli_inference import test, MODEL_PATH

    test(model_path=MODEL_PATH)


if __name__ == "__main__":
    app()
