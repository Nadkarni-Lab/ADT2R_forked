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

        # Write header for epoch summary
        summary_header = [
            "ep",
            "Train Loss", "Validation Loss", "Test Loss",
            "Val Accuracy", "Val Jaccard", "Val Recall", "Val WIS",
            "Test Accuracy", "Test Jaccard", "Test Recall", "Test WIS"
        ]
        summary_writer.writerow(summary_header)


        for ep in tqdm(range(args.total_epoch)):

            #tr_loss = train_one_epoch(model, train_loader, device, ep, train_writer)
            if args.return_q_values:
                tr_loss, train_q_values = train_one_epoch(model, train_loader, device, ep, train_writer,
                                                          return_q_values=True)
            else:
                tr_loss = train_one_epoch(model, train_loader, device, ep, train_writer, return_q_values=False)
                train_q_values = None

            #print(f"Epoch: {ep}, Train Loss: {tr_loss}")
            scheduler.step()

            # vl_loss, vl_acc, vl_jaccard, vl_recall, vl_wis = eval_(model, valid_loader, device, ep, val_writer)
            # ts_loss, ts_acc, ts_jaccard, ts_recall, ts_wis = eval_(model, test_loader, device, ep, test_writer)
            vl_loss, vl_acc, vl_jaccard, vl_recall, vl_wis, valid_q_values  = eval_(model, valid_loader, device, ep, val_writer)
            ts_loss, ts_acc, ts_jaccard, ts_recall, ts_wis, test_q_values = eval_(model, test_loader, device, ep, test_writer)

            #if ep >= 5:
                # vl_loss, vl_acc, vl_jaccard, vl_recall, vl_wis = eval_(model, valid_loader, device)
                # ts_loss, ts_acc, ts_jaccard, ts_recall, ts_wis = eval_(model, test_loader, device)
                # print(f"Epoch: {ep}, Train Loss: {tr_loss}, Validation Loss: {vl_loss}, Test Loss: {ts_loss}")
                # print(f"Validation Accuracy: {vl_acc}, Jaccard: {vl_jaccard}, Recall: {vl_recall}, WIS: {vl_wis}")
                # print(f"Test Accuracy: {ts_acc}, Jaccard: {ts_jaccard}, Recall: {ts_recall}, WIS: {ts_wis}")

            print(f"Epoch: {ep}, Train Loss: {tr_loss}, Validation Loss: {vl_loss}, Test Loss: {ts_loss}")
            print(f"Validation Accuracy: {vl_acc}, Jaccard: {vl_jaccard}, Recall: {vl_recall}, WIS: {vl_wis}")
            print(f"Test Accuracy: {ts_acc}, Jaccard: {ts_jaccard}, Recall: {ts_recall}, WIS: {ts_wis}")

            # Write epoch summary information to a separate file.
            summary_writer.writerow([
                ep,
                tr_loss, vl_loss, ts_loss,
                vl_acc, vl_jaccard, vl_recall, vl_wis,
                ts_acc, ts_jaccard, ts_recall, ts_wis
            ])

            # Save the Q-value arrays for this epoch
            # np.save(os.path.join(model_save_path, f"q_values_train_epoch_{ep}.npy"), train_q_values)
            # np.save(os.path.join(model_save_path, f"q_values_val_epoch_{ep}.npy"), valid_q_values)
            # np.save(os.path.join(model_save_path, f"q_values_test_epoch_{ep}.npy"), test_q_values)
            # Save training Q-values with gzip compression (compression level 9)

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

                # Log scalar summary stats.
                writer.add_scalar("TRAIN_Q_Values/Mean", q_mean, ep)
                writer.add_scalar("TRAIN_Q_Values/Std", q_std, ep)

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

                # Log scalar summary stats.
                writer.add_scalar("VALID_Q_Values/Mean", q_mean, ep)
                writer.add_scalar("VALID_Q_Values/Std", q_std, ep)

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

                # Log scalar summary stats.
                writer.add_scalar("TEST_Q_Values/Mean", q_mean, ep)
                writer.add_scalar("TEST_Q_Values/Std", q_std, ep)

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

    writer.close()

    print("Training completed")


