from cli.data_utils.data_providers.data_provider import DataProvider


TETHYS_ROOT_PATH = "/workspaces/depth_estimation_data/Tethys_computer_vision_dataset"
TETHYS_SUBPATH_TO_RGBS_FOLDER = "rigid_body_0/rgb/cam_0"
TETHYS_SUBPATH_TO_DEPTH_FOLDER = "rigid_body_0/depth/cam_0"


TethysDataProvider = DataProvider(name="tethys", root_path=TETHYS_ROOT_PATH, rgb_root=TETHYS_SUBPATH_TO_RGBS_FOLDER, depth_root=TETHYS_SUBPATH_TO_DEPTH_FOLDER)