import torch
import numpy as np
import argparse
from collections import OrderedDict
import os

# DiffusionReg imports
from utils.options import opts
from utils.diffusion_scheduler import DiffusionScheduler
from utils.se_math import se3
from modules.DCP.dcp import DCP

# Helper function to load point clouds
def load_point_cloud(filepath):
    """Loads a point cloud from a .npy file."""
    try:
        return np.load(filepath)
    except Exception as e:
        print(f"Error loading point cloud from {filepath}: {e}")
        return None

def init_opts(opts):
    """Initializes options for DiffusionReg."""
    opts.is_debug = False
    opts.is_test = True
    opts.schedule_type = "cosine"
    opts.n_diff_steps = 5
    opts.beta_1 = 0.2
    opts.beta_T = 0.8
    opts.sigma_r = 0.1
    opts.sigma_t = 0.01
    opts.is_add_noise = True # may need to change to False for deterministic registration
    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Corrected model parameters based on error messages
    opts.emb_nn = 'dgcnn' 
    opts.pointer = 'transformer'
    opts.head = 'svd'
    opts.emb_dims = 96  # Corrected from 512
    opts.n_blocks = 1
    opts.n_heads = 4
    opts.ff_dims = 256 # Corrected from 1024
    opts.dropout = 0.0
    return opts

def resample_pcd(pcd, n_points):
    """Resamples a point cloud to a specific number of points."""
    if len(pcd) > n_points:
        indices = np.random.choice(len(pcd), n_points, replace=False)
        pcd = pcd[indices]
    elif len(pcd) < n_points:
        padding = np.zeros((n_points - len(pcd), 3))
        pcd = np.vstack([pcd, padding])
    return pcd

def get_model(opts):
    """Loads the pretrained DiffusionReg model."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    opts.model_path = os.path.join(script_dir, "results/DiffusionReg-DiffusionDCP-tudl-diffusion_200_0.00010_0.05_0.05_0.03-nvids3_cosine/model_epoch19.pth")
    
    # DCP is the surrogate model used by DiffusionReg
    surrogate_model = DCP(opts)
    opts.vs = DiffusionScheduler(opts)

    try:
        # The model was trained with DataParallel, so keys might have a 'module.' prefix.
        state_dict = torch.load(opts.model_path, map_location=opts.device)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        surrogate_model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Could not load model weights: {e}")
        # Try loading without removing 'module.' prefix
        try:
            surrogate_model.load_state_dict(state_dict, strict=False)
        except Exception as e2:
            print(f"Failed to load model weights directly either: {e2}")
            return None

    surrogate_model = surrogate_model.to(opts.device)
    surrogate_model.eval()
    return surrogate_model

def register(model, opts, src_pcd, tgt_pcd):
    """
    Registers a source point cloud to a target point cloud.
    """
    with torch.no_grad():
        src_tensor = torch.from_numpy(src_pcd).float().unsqueeze(0).to(opts.device)
        tgt_tensor = torch.from_numpy(tgt_pcd).float().unsqueeze(0).to(opts.device)

        B = 1 # Batch size is 1
        H_t = torch.eye(4)[None].expand(B, -1, -1).to(opts.device)

        # The diffusion process loop from test.py
        for t in range(opts.n_diff_steps, 0, -1):
            # Apply current transformation
            src_t_unbatched = (H_t[0, :3, :3] @ src_tensor[0].T + H_t[0, :3, 3:4]).T
            src_t = src_t_unbatched.unsqueeze(0)

            # Create dummy normal vectors, as they are expected by the model
            src_normals = torch.zeros_like(src_t)
            tgt_normals = torch.zeros_like(tgt_tensor)

            # Predict transformation update
            Rs_pred, ts_pred = model.forward({
                "src_pcd": src_t,
                "model_pcd": tgt_tensor,
                "src_pcd_normal": src_normals,
                "model_pcd_normal": tgt_normals,
            })
            
            _delta_H_t = torch.cat([Rs_pred, ts_pred.unsqueeze(-1)], dim=2)
            delta_H_t = torch.eye(4)[None].expand(B, -1, -1).to(opts.device)
            delta_H_t[:, :3, :] = _delta_H_t
            H_0 = delta_H_t @ H_t

            if t > 1:
                # Interpolate for the next step of diffusion
                gamma0 = opts.vs.gamma0[t]
                gamma1 = opts.vs.gamma1[t]
                H_t = se3.exp(gamma0 * se3.log(H_0) + gamma1 * se3.log(H_t))
                
                # Add noise
                if opts.is_add_noise:
                    alpha_bar = opts.vs.alpha_bars[t]
                    alpha_bar_ = opts.vs.alpha_bars[t-1]
                    beta = opts.vs.betas[t]
                    cc = torch.sqrt(((1 - alpha_bar_) / (1. - alpha_bar_)) * beta)
                    scale = torch.cat([torch.ones(3) * opts.sigma_r, torch.ones(3) * opts.sigma_t])[None].to(opts.device)
                    noise = cc * scale * torch.randn(B, 6).to(opts.device)
                    H_noise = se3.exp(noise)
                    H_t = H_noise @ H_t
            else:
                 H_t = H_0


        # The final transformation
        final_transform = H_t[0].cpu().numpy()
        return final_transform

def main():
    parser = argparse.ArgumentParser(description="Register two point clouds using DiffusionReg.")
    parser.add_argument('source_pc', type=str, help='Path to the source point cloud (.npy file).')
    parser.add_argument('target_pc', type=str, help='Path to the target point cloud (.npy file).')
    args = parser.parse_args()

    # Initialize options
    global opts
    opts = init_opts(opts)

    # Load model
    model = get_model(opts)
    if model is None:
        print("Failed to load the model. Exiting.")
        return

    # Load point clouds
    src_pcd = load_point_cloud(args.source_pc)
    tgt_pcd = load_point_cloud(args.target_pc)
    if src_pcd is None or tgt_pcd is None:
        print("Failed to load point clouds. Exiting.")
        return

    # Ensure point clouds have the same number of points by sampling/padding
    n_points = 1024 # A common number of points for these models
    src_pcd = resample_pcd(src_pcd, n_points)
    tgt_pcd = resample_pcd(tgt_pcd, n_points)


    # Perform registration
    similarity_transform = register(model, opts, src_pcd, tgt_pcd)

    print("Computed Similarity Transformation:")
    print(similarity_transform)

if __name__ == '__main__':
    main()
