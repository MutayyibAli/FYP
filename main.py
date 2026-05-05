import torch  # Import PyTorch library
import time  # Import time library for measuring execution time
import os  # Import os library for file and directory operations
import sklearn  # Import scikit-learn for data splitting

from torch.optim import AdamW
from monai.losses import DiceCELoss

from nnu_model.nnu import nnUNetCore  # Your model from Task 3
from data_preprocessing.preprocessor import Preprocessor
from data_augmentation.data_loader import BraTSDataset
from utils.utils import get_patient_ids


def main():
    initial_setup()

    raw_path = ".raw"
    data_path = ".processed"
    pt_ids = get_patient_ids(raw_path)

    # Create directory for output of processed files if it does not exist.
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    data_dirs = [
        d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))
    ]
    out_dirs = [
        d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
    ]
    if len(data_dirs) != len(out_dirs):
        start = time.time()
        Preprocessor(raw_path, data_path, pt_ids).run()
        end = time.time()
        # 5 minutes
        print(f"Preprocessing time: {end - start} seconds")
    else:
        print("Preprocessed data already exists. Skipping preprocessing step.")

    train_ids, val_ids = sklearn.model_selection.train_test_split(
        pt_ids, test_size=0.2, random_state=123
    )

    training_dataset = BraTSDataset(data_path, train_ids, isTraining=True)
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=2, shuffle=True
    )
    validation_dataset = BraTSDataset(data_path, val_ids, isTraining=False)
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False
    )

    # Init Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = nnUNetCore(in_channels=4, out_channels=4).to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    num_epochs = 50
    print("Starting Training Loop with Metric Monitoring...")

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        epoch_loss = 0
        step = 0

        for batch_image, batch_label in train_loader:
            step += 1
            batch_image, batch_label = batch_image.to(device), batch_label.to(device)
            batch_label = batch_label.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_image)
            loss = loss_function(outputs, batch_label)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # --- VALIDATION PHASE (Metric Monitoring) ---
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        val_dice = 0
        val_iou = 0

        with torch.no_grad():  # Disable gradient calculation for speed and memory
            for val_image, val_label in val_loader:
                val_image, val_label = val_image.to(device), val_label.to(device)

                val_outputs = model(val_image)
                loss = loss_function(val_outputs, val_label)
                val_loss += loss.item()

                # Calculate metrics for the current batch
                dice, iou = calculate_metrics(val_outputs, val_label)
                val_dice += dice
                val_iou += iou

        # Calculate averages for the epoch
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        # Print Epoch Summary
        print(f"Epoch [{epoch + 1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(
            f"  Val Loss:   {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}"
        )
        print("-" * 50)

        # Save model checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"nnunet_brats_epoch_{epoch + 1}.pth")
            print("Model Checkpoint Saved!")


def calculate_metrics(predictions, labels, num_classes=4):
    """
    Calculates the Mean Dice Score and Mean Intersection over Union (IoU),
    ignoring the background class (0).
    """
    # Convert raw model logits to predicted classes (0, 1, 2, 3)
    preds = torch.argmax(predictions, dim=1)

    dice_scores = []
    iou_scores = []

    # Calculate metrics for each tumor class, ignoring background (0)
    for cls in range(1, num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls

        intersection = (pred_inds & target_inds).sum().float()
        union = pred_inds.sum() + target_inds.sum() - intersection

        # Only calculate if the class actually exists in the ground truth
        if target_inds.sum() > 0:
            dice = (2.0 * intersection) / (pred_inds.sum() + target_inds.sum() + 1e-8)
            iou = intersection / (union + 1e-8)

            dice_scores.append(dice.item())
            iou_scores.append(iou.item())

    # Average the scores across all present classes
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0

    return mean_dice, mean_iou


def initial_setup():
    # Enable TensorFloat-32 (TF32) for faster matrix operations on compatible NVIDIA GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# Make the main function run when the script is executed
if __name__ == "__main__":
    main()
