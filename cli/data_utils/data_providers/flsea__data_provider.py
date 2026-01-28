from cli.data_utils.data_providers.data_provider import DataProvider


FLSEA_ROOT_PATH = "/workspaces/depth_estimation_data/flsea_dp"
FLSEA_SUBPATH_TO_RGBS_FOLDER = "rgb"
FLSEA_SUBPATH_TO_DEPTH_FOLDER = "depth"


FLSeaDataProvider = DataProvider(name="flsea", root_path=FLSEA_ROOT_PATH, rgb_root=FLSEA_SUBPATH_TO_RGBS_FOLDER, depth_root=FLSEA_SUBPATH_TO_DEPTH_FOLDER)