import torch
import numpy as np
from tqdm import tqdm

import Load as ld
import Utils as ut
import Module as md

import csv


def train_one_epoch(model, loader, device, ep, csv_writer):
    model.train()
    loss_all = 0
    for bidx, batch in enumerate(loader):
        print("TRAIN bidx: ", bidx)

        record = dict()
        for key in batch.keys():
            record[key] = batch[key].to(device)

        ## changing update=True to is_train=True
        #loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b = model(record, update=True)
        loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b = model(record, is_train=True)
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

    return loss_all / len(loader)



def eval_(model, loader, device, ep, csv_writer):
    model.eval()
    loss_all = 0
    preds, reals, masks, probs, rewards = [], [], [], [], []

    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            #print("batch: ", batch)
            print("EVAL bidx: ", bidx)

            record = dict()
            for key in batch.keys():
                record[key]  = batch[key].to(device)

            #print("EVAL record: ", record)

            #loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b = model(record, update=False)
            loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, prob_b, pred_b = model(record, is_train=False)

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

    acc, jaccard, recall, wis = ut.calculate_metric(np.array(reals), np.array(preds), np.array(masks), np.array(probs), np.array(rewards))

    return loss_all / len(loader), acc, jaccard, recall, wis


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

    # Open CSV files for train, validation, and test losses.
    # The files will have the columns: bidx, ep, loss_act_b, loss_reg_b, loss_actor_b, loss_critic_b, loss_all
    with open(str(args.results_path)+'/train_loss.csv', 'w', newline='') as train_file, \
            open(str(args.results_path)+'/val_loss.csv', 'w', newline='') as val_file, \
            open(str(args.results_path)+'/test_loss.csv', 'w', newline='') as test_file, \
            open(str(args.results_path)+'/epoch_summary.csv', 'w', newline='') as summary_file:
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

            tr_loss = train_one_epoch(model, train_loader, device, ep, train_writer)
            #print(f"Epoch: {ep}, Train Loss: {tr_loss}")
            scheduler.step()

            vl_loss, vl_acc, vl_jaccard, vl_recall, vl_wis = eval_(model, valid_loader, device, ep, val_writer)
            ts_loss, ts_acc, ts_jaccard, ts_recall, ts_wis = eval_(model, test_loader, device, ep, test_writer)

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

    print("Training completed")


