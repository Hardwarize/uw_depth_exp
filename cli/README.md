export FEATURES_ROOT="/workspaces/uw_depth_exp/data_artifacts/features"
export SAMPLES_IDX_ROOT="/workspaces/uw_depth_exp/data_artifacts/samples"

python -m cli.cli list-dataproviders

python -m cli.cli check-dataprovider flsea

python3 -m cli.cli list-datasets <optional_dataprovider_name>
python3 -m cli.cli check-datasets <optional_dataprovider_name>
python3 -m cli.cli create-samples-idx-file
python3 -m cli.cli create-features
python3 -m cli.cli train