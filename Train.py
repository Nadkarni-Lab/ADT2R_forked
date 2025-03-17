import os

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import Load as ld
import Utils as ut
import Module as md
import h5py
import csv

import matplotlib.pyplot as plt
import numpy as np

# def plot_q_values_vs_rewards(q_values_batches, rewards_batches, epoch, save_path, writer=None):
#     """
#     q_values_batches: list of numpy arrays from each batch containing the Q-value for the taken action (v_hat_at)
#     rewards_batches: list of numpy arrays from each batch containing the corresponding rewards (record["reward"])
#     epoch: current epoch number (for labeling)
#     save_path: directory path to save the plot image
#     writer: TensorBoard SummaryWriter (optional)
#     """
#     # Concatenate arrays from all batches and flatten (assumes shape [B, T])
#     q_values_all = np.concatenate(q_values_batches, axis=0).flatten()
#     rewards_all = np.concatenate(rewards_batches, axis=0).flatten()
#
#     plt.figure(figsize=(6, 4))
#     plt.scatter(rewards_all, q_values_all, alpha=0.5)
#     plt.xlabel("Reward")
#     plt.ylabel("Predicted Q-value (v_hat_at)")
#     plt.title(f"Q-values vs Rewards at Epoch {epoch}")
#     plt.grid(True)
#
#     # Save to disk if a save_path is provided
#     if save_path:
#         plt.savefig(f"{save_path}/q_values_vs_rewards_epoch_{epoch}.png")
#
#     # Log to TensorBoard if a writer is provided
#     if writer:
#         writer.add_figure("Q-values_vs_Rewards", plt.gcf(), global_step=epoch)
#     plt.close()
#
#
# def plot_td_target_vs_qvalue(td_targets_batches, q_values_batches, epoch, save_path, writer=None):
#     """
#     (Optional) If you have computed TD targets per sample, you can use a similar function.
#     td_targets_batches: list of numpy arrays of computed TD targets per batch
#     q_values_batches: list of numpy arrays for the predicted Q-value for taken actions
#     """
#     td_targets_all = np.concatenate(td_targets_batches, axis=0).flatten()
#     q_values_all = np.concatenate(q_values_batches, axis=0).flatten()
#
#     plt.figure(figsize=(6, 4))
#     plt.scatter(q_values_all, td_targets_all, alpha=0.5)
#     plt.xlabel("Predicted Q-value (v_hat_at)")
#     plt.ylabel("TD Target")
#     plt.title(f"TD Targets vs Q-values at Epoch {epoch}")
#     plt.grid(True)
#
#     if save_path:
#         plt.savefig(f"{save_path}/td_target_vs_qvalue_epoch_{epoch}.png")
#     if writer:
#         writer.add_figure("TD_Target_vs_Q-value", plt.gcf(), global_step=epoch)
#     plt.close()


def plot_reward_vs_q_for_loader(model, loader, device, ep, save_path, writer, prefix):
    """
    Accumulates Q-values (v_hat_at) and rewards from the given loader,
    computes average values, plots and logs the results.

    prefix: a string to label the plot (e.g., "Train", "Validation", "Test")
    Returns: (avg_reward, avg_q)
    """
    q_values_batches = []
    rewards_batches = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            record = {k: batch[k].to(device) for k in batch.keys()}
            outputs = model(record, is_train=False)
            v_hat_at = outputs[7]  # Q-value for taken actions
            q_values_batches.append(v_hat_at.cpu().detach().numpy())
            rewards_batches.append(record["reward"].cpu().detach().numpy())

    q_values_all = np.concatenate(q_values_batches, axis=0).flatten()
    rewards_all = np.concatenate(rewards_batches, axis=0).flatten()
    avg_reward = np.mean(rewards_all)
    avg_q = np.mean(q_values_all)

    # Log the averages to TensorBoard with a prefix-specific key.
    writer.add_scalar(f"{prefix}/AverageReward", avg_reward, ep)
    writer.add_scalar(f"{prefix}/AverageQValue", avg_q, ep)

    # Create the scatter plot.
    plt.figure(figsize=(6, 4))
    plt.scatter(rewards_all, q_values_all, alpha=0.5)
    plt.xlabel("Reward")
    plt.ylabel("Predicted Q-value (v_hat_at)")
    plt.title(f"{prefix}: Q-values vs Rewards at Epoch {ep}")
    plt.grid(True)

    # Save plot to disk.
    plt.savefig(f"{save_path}/{prefix}_q_values_vs_rewards_epoch_{ep}.png")

    # Log the figure to TensorBoard.
    writer.add_figure(f"{prefix}/Q-values_vs_Rewards", plt.gcf(), global_step=ep)
    plt.close()

    return avg_reward, avg_q




class ADT2RDiscreteActionMatchEvaluator:
    def __init__(self, model, dataloader, device):
        """
        model: your ADT2R model instance.
        dataloader: a PyTorch DataLoader for the evaluation dataset.
        device: the device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in self.dataloader:
                # Move batch data to the proper device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Forward pass (is_train=False returns tuple with predicted actions at index 5)
                outputs = self.model(batch, is_train=False)
                action_preds = outputs[5]  # predicted actions tensor [B, T]
                true_actions = batch["action"]
                # Count matching actions
                correct += (action_preds.cpu() == true_actions.cpu()).sum().item()
                total += true_actions.numel()
        return 100.0 * correct / total if total > 0 else 0.0


def train_one_epoch(model, loader, device, ep, csv_writer, return_q_values=False):
    model.train()
    loss_all = 0
    train_q_values = []  # to collect Q-values for each batch

    for bidx, batch in enumerate(loader):
        print("TRAIN bidx: ", bidx)

        record = dict()
        for key in batch.keys():
            record[key] = batch[key].to(device)

        ## changing update=True to is_train=True
        #loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b = model(record, is_train=True)
        #loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b, q_values, selected_q_values = model(
        #                   record, is_train=True, return_q=False)
        if return_q_values:

            print("RETURN Q-values is TRUE")
            # Request Q-values from the model
            outputs = model(record, is_train=True, return_q=True)
            loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b, q_values, selected_q_values = outputs

            print("TRAIN loss_act_b: ", loss_act_b)
            print("TRAIN loss_reg_b: ", loss_reg_b)
            print("TRAIN loss_actor_b: ", loss_actor_b)
            print("TRAIN loss_critic_b: ", loss_critic_b)
            print("TRAIN prob_b: ", prob_b.shape)
            print("TRAIN pred_b: ", pred_b.shape)


            # Accumulate Q-values (e.g., the full Q-value tensor)
            train_q_values.append(q_values.cpu().detach().numpy())

        else:
            print("RETURN Q-values is FALSE")
            # Do not return Q-values
            loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b = model(record, is_train=True,
                                                                                        return_q=False)
            print("TRAIN loss_act_b: ", loss_act_b)
            print("TRAIN loss_reg_b: ", loss_reg_b)
            print("TRAIN loss_actor_b: ", loss_actor_b)
            print("TRAIN loss_critic_b: ", loss_critic_b)
            print("TRAIN prob_b: ", prob_b.shape)
            print("TRAIN pred_b: ", pred_b.shape)


        loss_all_b = loss_act_b + loss_reg_b + loss_actor_b + loss_critic_b
        print("TRAIN loss_all_b: ", loss_all_b)


        # Write batch info to CSV. We use .item() to get the scalar value.
        csv_writer.writerow([
            bidx,
            ep,
            loss_act_b.item(),
            loss_reg_b.item(),
            loss_actor_b.item(),
            loss_critic_b.item(),
            loss_all_b.item()
        ])

        loss_all += loss_all_b.cpu().detach().numpy()

    # return loss_all / len(loader)
    if return_q_values:
        # Concatenate Q-values from all batches (e.g., along the batch dimension)
        import numpy as np
        train_q_values = np.concatenate(train_q_values, axis=0)
        return loss_all / len(loader), train_q_values
    else:
        return loss_all / len(loader)



def eval_(model, loader, device, ep, csv_writer):
    model.eval()
    loss_all = 0
    preds, reals, masks, probs, rewards = [], [], [], [], []
    all_q_values = []  # to collect full Q-value tensors (v_hat) per batch

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            #print("batch: ", batch)
            print("EVAL bidx: ", bidx)

            record = dict()
            for key in batch.keys():
                record[key]  = batch[key].to(device)

            #print("EVAL record: ", record)

            #loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b = model(record, is_train=False)
            # Note: unpack additional outputs (q_values and selected_q_values)
            loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b, q_values, selected_q_values = model(
                record, is_train=False)

            print("EVAL loss_act_b: ", loss_act_b)
            print("EVAL loss_reg_b: ", loss_reg_b)
            print("EVAL loss_actor_b: ", loss_actor_b)
            print("EVAL loss_critic_b: ", loss_critic_b)
            print("EVAL prob_b: ", prob_b.shape)
            print("EVAL pred_b: ", pred_b.shape)

            loss_all_b = loss_act_b + loss_reg_b + loss_actor_b + loss_critic_b
            print("EVAL loss_all_b: ", loss_all_b)

            loss_all += loss_all_b.cpu().detach().numpy()


            # Write batch info to CSV file
            csv_writer.writerow([
                bidx,
                ep,
                loss_act_b.item(),
                loss_reg_b.item(),
                loss_actor_b.item(),
                loss_critic_b.item(),
                loss_all_b.item()
            ])

            preds.extend(pred_b.cpu().detach().numpy())
            reals.extend(batch["action"].cpu().detach().numpy())
            masks.extend(batch["seq_mask"].cpu().detach().numpy())
            probs.extend(prob_b.cpu().detach().numpy())
            rewards.extend(record["mortality"].cpu().detach().numpy())

            # Save the Q-values for this batch
            all_q_values.append(q_values.cpu().detach().numpy())

    # Optionally concatenate all batches along the batch dimension
    import numpy as np
    all_q_values = np.concatenate(all_q_values, axis=0)


    acc, jaccard, recall, wis = ut.calculate_metric(np.array(reals), np.array(preds), np.array(masks), np.array(probs), np.array(rewards))
    ### Also return all_q_values
    #return loss_all / len(loader), acc, jaccard, recall, wis
    return loss_all / len(loader), acc, jaccard, recall, wis, all_q_values


def eval_wMatchPercentage_(model, loader, device, ep, csv_writer):
    model.eval()
    loss_all = 0
    preds, reals, masks, probs, rewards = [], [], [], [], []
    all_q_values = []  # to collect full Q-value tensors (v_hat) per batch

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            print("EVAL bidx: ", bidx)
            record = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with evaluation mode. Note that index 5 is the predicted actions.
            loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b, q_values, selected_q_values = model(
                record, is_train=False)

            print("EVAL loss_act_b: ", loss_act_b)
            print("EVAL loss_reg_b: ", loss_reg_b)
            print("EVAL loss_actor_b: ", loss_actor_b)
            print("EVAL loss_critic_b: ", loss_critic_b)
            print("EVAL prob_b: ", prob_b.shape)
            print("EVAL pred_b: ", pred_b.shape)

            print("EVAL loss_all_b: ", loss_act_b + loss_reg_b + loss_actor_b + loss_critic_b)
            loss_all_b = loss_act_b + loss_reg_b + loss_actor_b + loss_critic_b
            loss_all += loss_all_b.cpu().detach().numpy()

            # Write batch-level loss metrics to CSV
            csv_writer.writerow([
                bidx,
                ep,
                loss_act_b.item(),
                loss_reg_b.item(),
                loss_actor_b.item(),
                loss_critic_b.item(),
                loss_all_b.item()
            ])

            # Collect predictions and true actions for discrete action match evaluation
            preds.extend(pred_b.cpu().detach().numpy())
            reals.extend(batch["action"].cpu().detach().numpy())
            masks.extend(batch["seq_mask"].cpu().detach().numpy())
            probs.extend(prob_b.cpu().detach().numpy())
            rewards.extend(record["mortality"].cpu().detach().numpy())

            # Save the Q-values for this batch
            all_q_values.append(q_values.cpu().detach().numpy())

    # Concatenate Q-values across batches
    import numpy as np
    all_q_values = np.concatenate(all_q_values, axis=0)

    # Calculate additional metrics using your utility function
    acc, jaccard, recall, wis = ut.calculate_metric(
        np.array(reals), np.array(preds), np.array(masks), np.array(probs), np.array(rewards)
    )

    # Compute the discrete action match percentage:
    reals_array = np.array(reals)
    preds_array = np.array(preds)
    match_percentage = 100.0 * (preds_array == reals_array).sum() / reals_array.size

    ## compute average reward:
    avg_reward = np.mean(np.array(rewards))

    # Return the loss, original metrics, the new action match metric, and the Q-values
    return loss_all / len(loader), acc, jaccard, recall, wis, match_percentage, all_q_values, avg_reward



def run(args, device, exp_name):
    """ Load datasets """
    print("**\t", exp_name)
    print("**\t Load dataset")

    ### add args.data as a dict to the args (other code used a ut.Namespace but we prob don't need that:
    args.data = {"train": args.train_data, "val": args.val_data, "test": args.test_data}

    #train_loader, valid_loader, test_loader = ld.load_fold(args)
    train_loader, valid_loader, test_loader = ld.load_fold_new(args)

    model = md.ADT2R(args.state_dim, args.action_dim, args.h_dim, args.n_heads, args.drop_p, args.max_timestep, device, args.lr, args.w_decay, args.lr_decay, args.lr_step, args.lam_actor, args.lam_critic, args.lam_reg, args.gamma, args.tau).to(device)
    scheduler = model.scheduler

    ## make a new directory using current run time stamp and save the model there.
    model_save_path = os.path.join(args.results_path, f"model_{exp_name}"+"_ADTR_Fold"+str(args.fold)+"/")
    os.makedirs(model_save_path, exist_ok=True)

    # Initialize the SummaryWriter with a log directory.
    writer = SummaryWriter(log_dir=str(args.q_log_dir)+f"model_{exp_name}"+"_ADTR_Fold"+str(args.fold))


    # Open CSV files for train, validation, and test losses.
    # The files will have the columns: bidx, ep, loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, loss_all
    with open(str(model_save_path)+'/train_loss.csv', 'w', newline='') as train_file, \
            open(str(model_save_path)+'/val_loss.csv', 'w', newline='') as val_file, \
            open(str(model_save_path)+'/test_loss.csv', 'w', newline='') as test_file, \
            open(str(model_save_path)+'/epoch_summary.csv', 'w', newline='') as summary_file:
        train_writer = csv.writer(train_file, delimiter='\t')
        val_writer = csv.writer(val_file, delimiter='\t')
        test_writer = csv.writer(test_file, delimiter='\t')
        summary_writer = csv.writer(summary_file, delimiter='\t')

        # Write header rows for batch-level logs.
        header = ["bidx", "ep", "loss_act_b", "loss_reg_b", "loss_actor_b", "loss_critic_b", "loss_all"]
        train_writer.writerow(header)
        val_writer.writerow(header)
        test_writer.writerow(header)

        # # Write header for epoch summary
        summary_header = [
            "ep",
            "Train Loss", "Validation Loss", "Test Loss",
            "Val Accuracy", "Val Jaccard", "Val Recall", "Val WIS", "Val Match",
            "Test Accuracy", "Test Jaccard", "Test Recall", "Test WIS", "Test Match",
            "Avg Train Actor Loss", "Avg Train Critic Loss",
            "Avg Valid Reward", "Avg Test Reward", "Avg Valid Q-value", "Avg Test Q-value"
        ]
        summary_writer.writerow(summary_header)

        # Lists to store historical averages (for cumulative plotting)
        train_avg_rewards = []
        train_avg_qs = []
        val_avg_rewards = []
        val_avg_qs = []
        test_avg_rewards = []
        test_avg_qs = []

        for ep in tqdm(range(args.total_epoch)):

            #tr_loss = train_one_epoch(model, train_loader, device, ep, train_writer)
            if args.return_q_values:
                tr_loss, train_q_values = train_one_epoch(model, train_loader, device, ep, train_writer,
                                                          return_q_values=True)
            else:
                tr_loss = train_one_epoch(model, train_loader, device, ep, train_writer, return_q_values=False)
                train_q_values = None


            # Lists to store per-batch losses for training metrics.
            actor_loss_epoch = []
            critic_loss_epoch = []

            # --- Training Phase ---
            for bidx, batch in enumerate(train_loader):
                record = {k: v.to(device) for k, v in batch.items()}

                # If return_q_values flag is set, we assume the model returns:
                # (action_loss, reg_loss, loss_actor, loss_critic, action_probs, action_preds, q_values, selected_q_values)
                outputs = model(record, is_train=True, return_q=True)
                loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b, q_values, selected_q_values = outputs

                # Append losses for aggregation.
                actor_loss_epoch.append(loss_actor_b.item())
                critic_loss_epoch.append(loss_critic_b.item())

            # Compute average training losses for this epoch.
            avg_actor_loss = np.mean(actor_loss_epoch)
            avg_critic_loss = np.mean(critic_loss_epoch)
            # Optionally, avg_td_error = np.sqrt(np.mean(np.square(td_error_epoch))
            # Log these to TensorBoard.
            writer.add_scalar("Train/ActorLoss", avg_actor_loss, ep)
            writer.add_scalar("Train/CriticLoss", avg_critic_loss, ep)

            #print(f"Epoch: {ep}, Train Loss: {tr_loss}")
            scheduler.step()

            # --- Evaluation Phase ---
            vl_loss, vl_acc, vl_jaccard, vl_recall, vl_wis, vl_match, valid_q_values, avg_valid_reward = eval_wMatchPercentage_(model, valid_loader, device, ep, val_writer)
            ts_loss, ts_acc, ts_jaccard, ts_recall, ts_wis, ts_match, test_q_values, avg_test_reward = eval_wMatchPercentage_(model, test_loader, device, ep, test_writer)


            print(f"Epoch: {ep}, Train Loss: {tr_loss}, Validation Loss: {vl_loss}, Test Loss: {ts_loss}")
            print(f"Validation Accuracy: {vl_acc}, Jaccard: {vl_jaccard}, Recall: {vl_recall}, WIS: {vl_wis}, Match: {vl_match}")
            print(f"Test Accuracy: {ts_acc}, Jaccard: {ts_jaccard}, Recall: {ts_recall}, WIS: {ts_wis}, Match: {ts_match}")

            # Plot reward vs Q-values for each loader and get the averages.
            train_avg_reward, train_avg_q = plot_reward_vs_q_for_loader(model, train_loader, device, ep,
                                                                        model_save_path, writer, "Train")
            val_avg_reward, val_avg_q = plot_reward_vs_q_for_loader(model, valid_loader, device, ep, model_save_path,
                                                                    writer, "Validation")
            test_avg_reward, test_avg_q = plot_reward_vs_q_for_loader(model, test_loader, device, ep, model_save_path,
                                                                      writer, "Test")

            # Store the averages for cumulative plotting.
            train_avg_rewards.append(train_avg_reward)
            train_avg_qs.append(train_avg_q)
            val_avg_rewards.append(val_avg_reward)
            val_avg_qs.append(val_avg_q)
            test_avg_rewards.append(test_avg_reward)
            test_avg_qs.append(test_avg_q)

            # Create a cumulative plot for each dataset: Average Q-value vs Average Reward over epochs.
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(train_avg_rewards, train_avg_qs, marker='o', label='Train')
            ax.plot(val_avg_rewards, val_avg_qs, marker='o', label='Validation')
            ax.plot(test_avg_rewards, test_avg_qs, marker='o', label='Test')
            ax.set_xlabel("Average Reward")
            ax.set_ylabel("Average Q-value")
            ax.set_title("Average Q-value vs Average Reward over Epochs")
            ax.legend()
            ax.grid(True)
            plt.savefig(f"{model_save_path}/cumulative_avg_reward_vs_avg_q_epoch_{ep}.png")
            writer.add_figure("Cumulative_Avg_Reward_vs_Avg_Q", fig, global_step=ep)
            plt.close()


            # If train_q_values is None, create an empty array; otherwise, process it.
            if train_q_values is None or len(train_q_values) == 0:
                train_q_values_arr = np.empty((0,), dtype=np.float32)
            else:
                try:
                    # Try concatenating the list of arrays
                    train_q_values_arr = np.concatenate(train_q_values, axis=0)
                except ValueError:
                    # If concatenation fails due to mismatched shapes, try stacking instead
                    train_q_values_arr = np.stack(train_q_values, axis=0)
                train_q_values_arr = train_q_values_arr.astype(np.float32)

                q_mean = train_q_values_arr.mean().item()
                q_std = train_q_values_arr.std().item()
                ##get median value of the Q-values
                q_median = np.median(train_q_values_arr)
                ## get min and max value of the Q-values
                q_min = train_q_values_arr.min().item()
                q_max = train_q_values_arr.max().item()

                # Log scalar summary stats.
                writer.add_scalar("TRAIN_Q_Values/Mean", q_mean, ep)
                writer.add_scalar("TRAIN_Q_Values/Median", q_median, ep)
                writer.add_scalar("TRAIN_Q_Values/Std", q_std, ep)
                writer.add_scalar("TRAIN_Q_Values/Min", q_min, ep)
                writer.add_scalar("TRAIN_Q_Values/Max", q_max, ep)

                # Log a histogram of the Q-values.
                writer.add_histogram("TRAIN_Q_Values/Histogram", train_q_values_arr, ep)

                print(f"Epoch {ep}: TRAIN Q-Values mean = {q_mean:.4f}, std = {q_std:.4f}")

            if valid_q_values is None or len(valid_q_values) == 0:
                valid_q_values_arr = np.empty((0,), dtype=np.float32)
            else:
                try:
                    valid_q_values_arr = np.concatenate(valid_q_values, axis=0)
                except ValueError:
                    valid_q_values_arr = np.stack(valid_q_values, axis=0)
                valid_q_values_arr = valid_q_values_arr.astype(np.float32)

                q_mean = valid_q_values_arr.mean().item()
                q_std = valid_q_values_arr.std().item()
                q_median = np.median(valid_q_values_arr)
                q_min = valid_q_values_arr.min().item()
                q_max = valid_q_values_arr.max().item()

                # Log scalar summary stats.
                writer.add_scalar("VALID_Q_Values/Mean", q_mean, ep)
                writer.add_scalar("VALID_Q_Values/Median", q_median, ep)
                writer.add_scalar("VALID_Q_Values/Std", q_std, ep)
                writer.add_scalar("VALID_Q_Values/Min", q_min, ep)
                writer.add_scalar("VALID_Q_Values/Max", q_max, ep)

                ## add percent valid matching
                writer.add_scalar("VALID_Q_Values/PercentActionsMatch", vl_match, ep)

                # Log a histogram of the Q-values.
                writer.add_histogram("VALID_Q_Values/Histogram", valid_q_values_arr, ep)

                print(f"Epoch {ep}: VALID Q-Values mean = {q_mean:.4f}, std = {q_std:.4f}")

            if test_q_values is None or len(test_q_values) == 0:
                test_q_values_arr = np.empty((0,), dtype=np.float32)
            else:
                try:
                    test_q_values_arr = np.concatenate(test_q_values, axis=0)
                except ValueError:
                    test_q_values_arr = np.stack(test_q_values, axis=0)
                test_q_values_arr = test_q_values_arr.astype(np.float32)

                q_mean = test_q_values_arr.mean().item()
                q_std = test_q_values_arr.std().item()
                q_median = np.median(test_q_values_arr)
                q_min = test_q_values_arr.min().item()
                q_max = test_q_values_arr.max().item()


                # Log scalar summary stats.
                writer.add_scalar("TEST_Q_Values/Mean", q_mean, ep)
                writer.add_scalar("TEST_Q_Values/Median", q_median, ep)
                writer.add_scalar("TEST_Q_Values/Std", q_std, ep)
                writer.add_scalar("TEST_Q_Values/Min", q_min, ep)
                writer.add_scalar("TEST_Q_Values/Max", q_max, ep)

                ## add percent test matching
                writer.add_scalar("TEST_Q_Values/PercentActionsMatch", ts_match, ep)

                # Log a histogram of the Q-values.
                writer.add_histogram("TEST_Q_Values/Histogram", test_q_values_arr, ep)

                print(f"Epoch {ep}: TEST Q-Values mean = {q_mean:.4f}, std = {q_std:.4f}")


            ### SAVE Q-Values to h5 files only every 10 epochs
            if ep % 10 == 0:
                # If train_q_values is None, create an empty array; otherwise, process it.
                if train_q_values is None or len(train_q_values) == 0:
                    train_q_values_arr = np.empty((0,), dtype=np.float32)
                else:
                    try:
                        # Try concatenating the list of arrays
                        train_q_values_arr = np.concatenate(train_q_values, axis=0)
                    except ValueError:
                        # If concatenation fails due to mismatched shapes, try stacking instead
                        train_q_values_arr = np.stack(train_q_values, axis=0)
                    train_q_values_arr = train_q_values_arr.astype(np.float32)

                with h5py.File(os.path.join(model_save_path, f"q_values_train_epoch_{ep}.h5"), 'w') as f:
                    f.create_dataset('q_values', data=train_q_values_arr, compression='gzip', compression_opts=9)

                # Save validation Q-values
                # If valid_q_values is None, create an empty array; otherwise, process it.
                if valid_q_values is None or len(valid_q_values) == 0:
                    valid_q_values_arr = np.empty((0,), dtype=np.float32)
                else:
                    try:
                        # Try concatenating the list of arrays
                        valid_q_values_arr = np.concatenate(valid_q_values, axis=0)
                    except ValueError:
                        # If concatenation fails due to mismatched shapes, try stacking instead
                        valid_q_values_arr = np.stack(valid_q_values, axis=0)
                    valid_q_values_arr = valid_q_values_arr.astype(np.float32)

                with h5py.File(os.path.join(model_save_path, f"q_values_val_epoch_{ep}.h5"), 'w') as f:
                    f.create_dataset('q_values', data=valid_q_values_arr, compression='gzip', compression_opts=9)

                # Save test Q-values
                # If test_q_values is None, create an empty array; otherwise, process it.
                if test_q_values is None or len(test_q_values) == 0:
                    test_q_values_arr = np.empty((0,), dtype=np.float32)
                else:
                    try:
                        # Try concatenating the list of arrays
                        test_q_values_arr = np.concatenate(test_q_values, axis=0)
                    except ValueError:
                        # If concatenation fails due to mismatched shapes, try stacking instead
                        test_q_values_arr = np.stack(test_q_values, axis=0)
                    test_q_values_arr = test_q_values_arr.astype(np.float32)


                with h5py.File(os.path.join(model_save_path, f"q_values_test_epoch_{ep}.h5"), 'w') as f:
                    f.create_dataset('q_values', data=test_q_values_arr, compression='gzip', compression_opts=9)

            ## Summary writer for the epoch
            # Write epoch summary to CSV
            summary_writer.writerow([
                ep,
                tr_loss, vl_loss, ts_loss,
                vl_acc, vl_jaccard, vl_recall, vl_wis, vl_match,
                ts_acc, ts_jaccard, ts_recall, ts_wis, ts_match,
                avg_actor_loss, avg_critic_loss,
                avg_valid_reward, avg_test_reward, np.mean(valid_q_values_arr), np.mean(test_q_values_arr)
            ])

    writer.close()

    print("Training completed")


