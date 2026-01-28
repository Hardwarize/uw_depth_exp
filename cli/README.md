export FEATURES_ROOT="/workspaces/uw_depth_exp/data_artifacts/features"
export SAMPLES_IDX_ROOT="/workspaces/uw_depth_exp/data_artifacts/samples"

python -m cli.cli list-dataproviders

python -m cli.cli check-dataprovider flsea

python3 -m cli.cli list-datasets <optional_dataprovider_name>
python3 -m cli.cli check-datasets <optional_dataprovider_name>
python3 -m cli.cli create-samples-idx-file
python3 -m cli.cli create-features
python3 -m cli.cli check-features-random-image

python3 -m cli.cli train -train flsea:flsea__canyons__flatiron,flsea__canyons__horse_canyon,flsea__canyons__tiny_canyon,flsea__red_sea__big_dice_loop,flsea__red_sea__coral_table_loop,flsea__red_sea__cross_pyramid_loop,flsea__red_sea__dice_path,flsea__red_sea__landward_path,flsea__red_sea__northeast_path,flsea__red_sea__pier_path -val flsea:flsea__canyons__u_canyon,flsea__red_sea__sub_pier