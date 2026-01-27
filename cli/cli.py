import typer
from typing import Optional
from typing_extensions import Annotated

from cli.data_utils.data_providers import DataProviders
from cli.data_utils.depth_dataset import DepthDataset
from cli.cli_train import train_UDFNet

app = typer.Typer()


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
                    ds = DepthDataset(
                        provider_name = dp.name,
                        provider_root_path = dp.root_path,
                        dataset_name = dataset,
                        rgb_subpath = dp.rgb_root,
                        depth_subpath = dp.depth_root
                    )
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
                    ds = DepthDataset(
                        provider_name = dp.name,
                        provider_root_path = dp.root_path,
                        dataset_name = dataset,
                        rgb_subpath = dp.rgb_root,
                        depth_subpath = dp.depth_root
                    )
                    ds.create_samples_idx_csv()
            except Exception as e:
                print(f" - Error listing datasets: {e}")
        return


@app.command()
def create_features(name: Annotated[Optional[str], typer.Argument()] = None):
    
    if name is None:
        print("Listing datasets for all providers:")
        for dp in DataProviders:
            print(f"Provider: {dp.name}")
            try:
                datasets = dp.list_datasets()
                for dataset in datasets:
                    ds = DepthDataset(
                        provider_name = dp.name,
                        provider_root_path = dp.root_path,
                        dataset_name = dataset,
                        rgb_subpath = dp.rgb_root,
                        depth_subpath = dp.depth_root
                    )
                    ds.create_features()
            except Exception as e:
                print(f" - Error listing datasets: {e}")
        return


@app.command()
def train():
    train_UDFNet()

if __name__ == "__main__":
    app()
