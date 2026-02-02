import os

import torch
from torch.utils.data import DataLoader

import numpy as np

from depth_estimation.model.model import UDFNet
from depth_estimation.utils.loss import RMSELoss, SILogLoss, MARELoss

from cli.data_utils.model_dataset.model_dataset import get_model_dataset


############################################################
###################### CONFIG ##############################
############################################################

BATCH_SIZE = 32
DEVICE = "cuda"

#MODEL_PATH = "data/saved_models/model_e22_udfnet_lr0.0001_bs6_lrd0.9.pth"
MODEL_PATH = "/workspaces/depth_estimation_data/training_artifacts/trial_01/saved_models/model_e10_udfnet_lr0.0001_bs8_lrd0.9.pth"

############################################################
############################################################
############################################################


# losses
rmse_lin = RMSELoss()
rmse_log = SILogLoss(correction=0.0, scaling=1.0)
rmse_silog = SILogLoss(correction=1.0, scaling=1.0)
mare = MARELoss()

# max depth
dmax = []


@torch.no_grad()
#def test(model_path=MODEL_PATH, datasets_samples_files = ['/workspaces/uw_depth_exp/data_artifacts/samples/Repmus_motorbike/samples.csv'], trial_name: str = 'test'):
#def test(model_path=MODEL_PATH, datasets_samples_files = ['/workspaces/uw_depth_exp/data_artifacts/samples/Orsted_encirclement/samples.csv'], trial_name: str = 'test'):
#def test(model_path=MODEL_PATH, datasets_samples_files = ['/workspaces/uw_depth_exp/data_artifacts/samples/Oceanpact_box/samples.csv'], trial_name: str = 'test'):
def test(model_path=MODEL_PATH, datasets_samples_files = ['/workspaces/uw_depth_exp/data_artifacts/samples/Limknow_drutenpv3/samples.csv'], trial_name: str = 'test'):
    #def test(model_path=MODEL_PATH, datasets_samples_files = ['/workspaces/uw_depth_exp/data_artifacts/samples/Limknow_drutenpv2/samples.csv'], trial_name: str = 'test'):
    # device info
    print(f"Using device {DEVICE}")

    output_rgb_files = []
    output_predictions = []
    output_masked_predictions = []
    output_rgb_paths = []
    output_rgb_inputs = []
    output_gt_depths = [] 

    #if trial_name and not os.getenv("RESULT_ARTIFACTS_ROOT"):
    #    print(f"If trial_name is provided, RESULT_ARTIFACTS_ROOT env var must be set.")

    if model_path == MODEL_PATH:
        print(f"Warning! Using default model path: {MODEL_PATH}")
    
    # model
    print(f"Loading model from {model_path}")
    model = UDFNet(n_bins=80).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loading model done.")

    dataset = get_model_dataset(samples_idx_file=datasets_samples_files, shuffle=False)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, drop_last=False)

    n_batches = len(dataloader)
    ranges = [None, 5.0, 1.0]
    rmse_lin_losses = [[], [], []]
    rmse_log_losses = [[], [], []]
    rmse_silog_losses = [[], [], []]
    mare_losses = [[], [], []]
    for batch_id, data in enumerate(dataloader):

        # inputs
        rgb = data[0].to(DEVICE)  # RGB image
        target = data[1].to(DEVICE)  # depth image
        mask = data[2].to(DEVICE)  # mask for valid values
        prior = data[3].to(DEVICE)  # precomputed features and depth values
        rgb_paths = data[4]  # paths of rgb images

        dmax.append(target.max().item())

        # nullprior
        # prior[:, :, :, :] = 0.0

        # outputs
        prediction, _ = model(rgb, prior)

        # # enforce best fit at keypoint locations
        # priors_mask = (prior[:,1,:,:] == 1).unsqueeze(1)  # mask for location of keypoints, probability is ==1 at keypoint locations
        # scales = torch.zeros(BATCH_SIZE, 1, 1, 1).to(DEVICE)  # retrieve correction scale for best scale in RMSE sense
        # for i in range(BATCH_SIZE):
        #     scales[i] = get_scale(prediction[i], target[i], priors_mask[i])
        # prediction = scales * prediction
        if trial_name:
            for i, pred in enumerate(prediction):
                output_rgb_files.append(rgb_paths[i])
                output_predictions.append(pred.cpu().numpy())
                output_masked_predictions.append(pred[mask[i]].cpu().numpy())
                output_masked_predictions.append(target[i][mask[i]].cpu().numpy())
                output_rgb_paths.append(rgb_paths[i])
                output_rgb_inputs.append(rgb[i].cpu().numpy())
                output_gt_depths.append(target[i].cpu().numpy())

        # loss
        for i, r in enumerate(ranges):

            # for whole range select all
            if r is None:
                m_target = target[mask]
                m_prediction = prediction[mask]

            # for finite range select pixels with mask
            else:

                # mask for this range
                range_mask = target[mask] < r

                # skip if mask is empty
                if not range_mask.any():
                    continue

                m_target = target[mask][range_mask]
                m_prediction = prediction[mask][range_mask]

            # loss
            rmse_lin_losses[i].append(rmse_lin(m_prediction, m_target).item())
            rmse_log_losses[i].append(rmse_log(m_prediction, m_target).item())
            rmse_silog_losses[i].append(rmse_silog(m_prediction, m_target).item())
            mare_losses[i].append(mare(m_prediction, m_target).item())

        if batch_id % 10 == 0:
            print(f"{batch_id}/{n_batches}")

    for i, r in enumerate(ranges):
        print(f"Range: {r} m, using MEAN reduction:")
        print(f"RMSE (lin): {np.nanmean(rmse_lin_losses[i])}")
        print(f"RMSE (log): {np.nanmean(rmse_log_losses[i])}")
        print(f"RMSE (silog): {np.nanmean(rmse_silog_losses[i])}")
        print(f"MARE: {np.nanmean(mare_losses[i])}")
        print("---")
        print(f"Range: {r} m, using MEDIAN reduction:")
        print(f"RMSE (lin): {np.nanmedian(rmse_lin_losses[i])}")
        print(f"RMSE (log): {np.nanmedian(rmse_log_losses[i])}")
        print(f"RMSE (silog): {np.nanmedian(rmse_silog_losses[i])}")
        print(f"MARE: {np.nanmedian(mare_losses[i])}")
        print("\n===\n")

    print(f"max_depth mean: {np.mean(dmax)}")
    print(f"max_depth median: {np.median(dmax)}")

    if trial_name:
        np.savez_compressed(
            'Limknow_drutenpv3_rbf.npz',
            filenames=np.array(output_rgb_files),
            preds=np.stack(output_predictions),
            #masked=np.stack(output_masked_predictions),
            gts=np.stack(output_gt_depths),
            rgbs=np.stack(output_rgb_inputs),
            rgb_paths=np.array(output_rgb_paths)
        )


def get_scale(prediction, target, mask):
    
    # A*s = B
    A = prediction[mask].unsqueeze(1)
    B = target[mask].unsqueeze(1)
    s = torch.linalg.lstsq(A, B).solution

    return s


if __name__ == "__main__":
    test()
