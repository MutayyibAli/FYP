import os
import numpy as np
import torch
import nibabel as nib
import monai
import scipy.ndimage
import itertools
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# =====================================================================
# 1. Model Architecture (From Task 3)
# =====================================================================
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            stride=stride,
        )
        self.norm1 = torch.nn.InstanceNorm3d(out_channels)
        self.relu1 = torch.nn.LeakyReLU(inplace=True)
        self.conv2 = torch.nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = torch.nn.InstanceNorm3d(out_channels)
        self.relu2 = torch.nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu2(self.norm2(self.conv2(self.relu1(self.norm1(self.conv1(x))))))


class nnUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 4  # T1, T1ce, T2, FLAIR
        out_channels = 4  # Background, Necrotic Core, Edema, Enhancing Tumor

        # --- ENCODER ---
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64, 2)
        self.enc3 = ConvBlock(64, 128, 2)
        self.enc4 = ConvBlock(128, 256, 2)
        self.enc5 = ConvBlock(256, 320, 2)
        self.bottleneck = ConvBlock(320, 320, 2)

        # --- DECODER ---
        self.up5 = torch.nn.ConvTranspose3d(320, 320, kernel_size=2, stride=2)
        self.dec5 = ConvBlock(640, 320)
        self.up4 = torch.nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = torch.nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = torch.nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = torch.nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.final_conv = torch.nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        d5 = self.dec5(torch.cat([self.up5(b), e5], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        f = self.final_conv(d1)
        return f


# =====================================================================
# 2. Inference Utilities (From Task 4)
# =====================================================================
def run_sliding_window(input_tensor, model):
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*non-tuple sequence.*"
    )
    with torch.amp.autocast(
        device_type=input_tensor.device.type,
        dtype=torch.float16 if input_tensor.device.type == "cuda" else torch.bfloat16,
    ):
        logits = monai.inferers.sliding_window_inference(
            inputs=input_tensor,
            roi_size=(128, 128, 128),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode="gaussian",
        )
    return torch.softmax(logits, dim=1)


def tta_sliding_window_inference(input_tensor, model):
    spatial_dims = (2, 3, 4)
    flip_combinations = []
    for r in range(len(spatial_dims) + 1):
        flip_combinations.extend(itertools.combinations(spatial_dims, r))

    accumulated_probs = None
    for axes in flip_combinations:
        flipped_input = (
            torch.flip(input_tensor, dims=axes) if len(axes) > 0 else input_tensor
        )
        probs = run_sliding_window(flipped_input, model)
        unflipped_probs = torch.flip(probs, dims=axes) if len(axes) > 0 else probs

        if accumulated_probs is None:
            accumulated_probs = unflipped_probs
        else:
            accumulated_probs += unflipped_probs

    avg_probs = accumulated_probs / len(flip_combinations)
    return avg_probs


# =====================================================================
# 3. Post-Processing Utilities (From Task 4)
# =====================================================================
def get_largest_connected_component(mask):
    if not np.any(mask):
        return mask
    labeled_array, num_features = scipy.ndimage.label(mask)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labeled_array.ravel())
    largest_component_label = component_sizes[1:].argmax() + 1
    return (labeled_array == largest_component_label).astype(np.uint8)


def apply_morphological_refinement(mask, iterations=1):
    mask = mask.astype(bool)
    mask = scipy.ndimage.binary_opening(mask, iterations=iterations)
    mask = scipy.ndimage.binary_closing(mask, iterations=iterations)
    return mask.astype(np.uint8)


def postprocess_prediction(pred_mask):
    if hasattr(pred_mask, "cpu"):
        pred_mask = pred_mask.cpu().numpy()
    processed_mask = np.zeros_like(pred_mask)
    for class_id in np.unique(pred_mask):
        if class_id == 0:
            continue
        class_mask = (pred_mask == class_id).astype(np.uint8)
        class_mask = get_largest_connected_component(class_mask)
        class_mask = apply_morphological_refinement(class_mask)
        processed_mask[class_mask > 0] = class_id
    return processed_mask


# =====================================================================
# 4. Main Execution Logic
# =====================================================================
def save_layered_slice(bg_slice, mask_slice, save_path):
    """Helper function to replicate the project's layered tumor plotting and save it."""
    cmap_wt = ListedColormap(["yellow"])
    cmap_tc = ListedColormap(["orange"])
    cmap_et = ListedColormap(["red"])

    # Create masks
    wt_mask = np.ma.masked_where(mask_slice == 0, np.ones_like(mask_slice))
    tc_mask = np.ma.masked_where(
        (mask_slice != 1) & (mask_slice != 3), np.ones_like(mask_slice)
    )
    et_mask = np.ma.masked_where(mask_slice != 3, np.ones_like(mask_slice))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(bg_slice, cmap="gray")
    ax.imshow(wt_mask, cmap=cmap_wt, alpha=0.5)
    ax.imshow(tc_mask, cmap=cmap_tc, alpha=0.5)
    ax.imshow(et_mask, cmap=cmap_et, alpha=0.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_simple_slice(img_slice, save_path):
    """Helper function to save raw modality images."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_slice, cmap="gray")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def save_black_slice(shape, save_path):
    """Saves a completely black image of the specified shape."""
    fig, ax = plt.subplots(figsize=(6, 6))
    black_img = np.zeros(shape)
    ax.imshow(black_img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def get_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    uploads_dir = "uploads"
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)

    # 1. Load 4 modalities
    t1_img = (
        nib.load(os.path.join(uploads_dir, "t1.nii.gz")).get_fdata().astype(np.float32)
    )
    t1ce_img = (
        nib.load(os.path.join(uploads_dir, "t1ce.nii.gz"))
        .get_fdata()
        .astype(np.float32)
    )
    t2_img = (
        nib.load(os.path.join(uploads_dir, "t2.nii.gz")).get_fdata().astype(np.float32)
    )
    flair_img = (
        nib.load(os.path.join(uploads_dir, "flair.nii.gz"))
        .get_fdata()
        .astype(np.float32)
    )

    # 2. Check for ground truth
    ground_truth_path = os.path.join(uploads_dir, "ground_truth.nii.gz")
    ground_truth_provided = os.path.exists(ground_truth_path)
    if ground_truth_provided:
        true_mask = nib.load(ground_truth_path).get_fdata().astype(np.uint8)
        true_mask[true_mask == 4] = 3  # Mapping standard from the project
    else:
        true_mask = np.zeros_like(t1_img, dtype=np.uint8)  # Dummy mask

    # 3. Preprocess Data (Stack and Z-score normalization per project)
    # Order required by project: [t1, t1ce, t2, flair]
    input_tensor = np.stack([t1_img, t1ce_img, t2_img, flair_img], axis=0)
    for i in range(4):
        input_tensor[i] = (input_tensor[i] - np.mean(input_tensor[i])) / (
            np.std(input_tensor[i]) + 1e-8
        )

    input_tensor = (
        torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    )

    # 4. Load Pretrained Model
    model = nnUNet().to(device)
    checkpoint_path = "training/model.pth"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Using model_state_dict as saved in the project's checkpoint function
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # 5. Run inference with TTA
    with torch.no_grad():
        avg_probs = tta_sliding_window_inference(input_tensor=input_tensor, model=model)
        pred_mask = torch.argmax(avg_probs, dim=1).squeeze(0).cpu().numpy()

    refined_pred_mask = postprocess_prediction(pred_mask)

    # 6. Visualization - Find the best 2D slice
    if ground_truth_provided:
        tumor_area_per_slice = np.sum(true_mask > 0, axis=(0, 1))
    else:
        tumor_area_per_slice = np.sum(refined_pred_mask > 0, axis=(0, 1))

    best_slice_idx = np.argmax(tumor_area_per_slice)
    if tumor_area_per_slice[best_slice_idx] == 0:
        best_slice_idx = true_mask.shape[2] // 2  # Fallback to middle

    # Extract slices and rotate 90 degrees (as done in project)
    t1_slice = np.rot90(t1_img[:, :, best_slice_idx])
    t1ce_slice = np.rot90(t1ce_img[:, :, best_slice_idx])
    t2_slice = np.rot90(t2_img[:, :, best_slice_idx])
    flair_slice = np.rot90(flair_img[:, :, best_slice_idx])

    true_mask_slice = np.rot90(true_mask[:, :, best_slice_idx])
    pred_mask_slice = np.rot90(refined_pred_mask[:, :, best_slice_idx])

    # Save Individual Images to Static Folder
    save_simple_slice(
        t1_slice,
        os.path.join(static_dir, "t1.png"),
    )
    save_simple_slice(
        t1ce_slice,
        os.path.join(static_dir, "t1ce.png"),
    )
    save_simple_slice(
        t2_slice,
        os.path.join(static_dir, "t2.png"),
    )
    save_simple_slice(
        flair_slice,
        os.path.join(static_dir, "flair.png"),
    )

    # Layered plots use FLAIR as background according to the project PDF
    save_layered_slice(
        flair_slice,
        pred_mask_slice,
        os.path.join(static_dir, "prediction.png"),
    )

    if ground_truth_provided:
        save_layered_slice(
            flair_slice,
            true_mask_slice,
            os.path.join(static_dir, "ground_truth.png"),
        )
    else:
        save_black_slice(
            flair_slice.shape, os.path.join(static_dir, "ground_truth.png")
        )
