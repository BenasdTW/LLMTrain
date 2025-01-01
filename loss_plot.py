import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_log_file(file_path):
    losses = {
        "train": {"steps": [], "losses": []},
        "eval": {"steps": [], "losses": []},
    }
    try:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                if "loss" in data and "eval_loss" not in data:
                    losses["train"]["steps"].append(data["step"])
                    losses["train"]["losses"].append(data["loss"])
                elif "eval_loss" in data:
                    losses["eval"]["steps"].append(data["step"])
                    losses["eval"]["losses"].append(data["eval_loss"])
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found. Skipping.")
    return losses


def plot_in_grid(loss_dicts, file_names, size=(2, 3)):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(size[0], size[1], figsize=(18, 10))  # Create a 2x3 grid of subplots
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for idx, (loss_dict, file_name) in enumerate(zip(loss_dicts, file_names)):
        # Prepare the data for Seaborn
        train_df = pd.DataFrame({
            "Step": loss_dict["train"]["steps"],
            "Loss": loss_dict["train"]["losses"],
            "Type": "Training",
        })
        eval_df = pd.DataFrame({
            "Step": loss_dict["eval"]["steps"],
            "Loss": loss_dict["eval"]["losses"],
            "Type": "Evaluation",
        })
        combined_df = pd.concat([train_df, eval_df])

        # Plot on the respective subplot
        sns.lineplot(data=combined_df, x="Step", y="Loss", hue="Type", ax=axes[idx])
        axes[idx].set_title(file_name)
        axes[idx].set_xlabel("Steps")
        axes[idx].set_ylabel("Loss")
        axes[idx].legend(title="Loss Type")

        # Save the individual plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=combined_df, x="Step", y="Loss", hue="Type")
        plt.title(file_name)
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend(title="Loss Type")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./plots/{file_name.replace('.log', '')}_loss_plot.png")
        plt.close()  # Close the individual plot to free memory

    # Hide any unused subplots (if fewer than 6 files)
    for i in range(len(loss_dicts), len(axes)):
        fig.delaxes(axes[i])

    # Display the grid of plots
    plt.tight_layout()
    plt.show()


# Main script to parse logs and plot losses
# log_file_names = [
#     "text2sql-1b.log",
#     "text2sql-3b.log",
#     "text2sql-8b.log",
#     "text2sql-1b-Instruct.log",
#     "text2sql-3b-Instruct.log",
#     "text2sql-8b-Instruct.log",
# ]

# log_file_names = [
#     "text2sql-1b-Instruct-2.log",
#     "text2sql-3b-Instruct-2.log",
#     "text2sql-8b-Instruct-2.log",
#     "text2sql-1b-Instruct-format.log",
#     "text2sql-3b-Instruct-format.log",
#     "text2sql-8b-Instruct-format.log",
# ]

log_file_names = [
    "text2sql-1b-Instruct-less.log",
    "text2sql-1b-Instruct-loraplus.log",
    "text2sql-1b-Instruct-extra.log",
    "text2sql-3b-Instruct-loraplus-extra.log",
]

log_dir = "./logs/"
output_dir = "./plots/"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

loss_dicts = []

for file_name in log_file_names:
    file_path = os.path.join(log_dir, file_name)
    loss_dicts.append(parse_log_file(file_path))

# Plot all losses in a 2x3 grid and save individual plots
plot_in_grid(loss_dicts, log_file_names, (2, 2))
