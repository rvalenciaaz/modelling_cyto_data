import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_FOLDER = "replication_files"

def main():
    cv_losses_file = os.path.join(OUTPUT_FOLDER, "cv_fold_losses.pkl")
    final_losses_file = os.path.join(OUTPUT_FOLDER, "final_losses.pkl")

    # ------------------------------------------------------------------
    # 1. Cross-validation fold losses
    # ------------------------------------------------------------------
    if not os.path.exists(cv_losses_file):
        print(f"File {cv_losses_file} not found. Cannot replicate fold-loss plots.")
    else:
        with open(cv_losses_file, "rb") as f:
            cv_data = pickle.load(f)
            fold_train_losses = cv_data["train_losses"]  # list of lists
            fold_val_losses   = cv_data["val_losses"]    # list of lists

        # 1a) Single figure overlaying all folds' train/val losses
        plt.figure(figsize=(10, 6))
        for i in range(len(fold_train_losses)):
            epochs_i = range(1, len(fold_train_losses[i]) + 1)
            plt.plot(epochs_i, fold_train_losses[i], label=f"Train Fold {i+1}")
            plt.plot(epochs_i, fold_val_losses[i],   label=f"Val Fold {i+1}")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("All Folds Train & Val Loss Curves (Replicated)")
        plt.legend(loc="upper right", ncol=2, fontsize="small")
        plt.grid(True)
        plt.tight_layout()
        all_folds_path = os.path.join(OUTPUT_FOLDER, "replicated_cv_folds_loss_trends.png")
        plt.savefig(all_folds_path)
        plt.close()
        print(f"Recreated single plot with all fold lines => {all_folds_path}")

        # 1b) Aggregated mean ± std across folds
        all_train = np.array(fold_train_losses)  # shape: (n_folds, n_epochs)
        all_val   = np.array(fold_val_losses)    # shape: (n_folds, n_epochs)
        mean_train = all_train.mean(axis=0)
        std_train  = all_train.std(axis=0)
        mean_val   = all_val.mean(axis=0)
        std_val    = all_val.std(axis=0)

        epochs = range(1, all_train.shape[1] + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, mean_train, label="Train Loss (mean)")
        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.3)
        plt.plot(epochs, mean_val, label="Val Loss (mean)")
        plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Aggregated CV Loss (Mean ± Std) (Replicated)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        agg_plot_path = os.path.join(OUTPUT_FOLDER, "replicated_cv_aggregated_loss.png")
        plt.savefig(agg_plot_path)
        plt.close()
        print(f"Recreated aggregated CV plot => {agg_plot_path}")

    # ------------------------------------------------------------------
    # 2. Final training losses (entire training set)
    # ------------------------------------------------------------------
    if not os.path.exists(final_losses_file):
        print(f"File {final_losses_file} not found. Cannot replicate final training-loss plot.")
    else:
        with open(final_losses_file, "rb") as f:
            final_losses = pickle.load(f)  # list of losses
        epochs_final = range(1, len(final_losses) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_final, final_losses, label='Final Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Final Model Training Loss vs. Epochs (Replicated)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        final_plot_path = os.path.join(OUTPUT_FOLDER, "replicated_final_training_loss.png")
        plt.savefig(final_plot_path)
        plt.close()
        print(f"Recreated final training loss plot => {final_plot_path}")

    print("\nDone replicating plots.")

if __name__ == "__main__":
    main()

