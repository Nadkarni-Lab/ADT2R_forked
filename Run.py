import argparse, torch, os

import Train as tr
import Utils as ut


def parse_args():
    parser = argparse.ArgumentParser("ADTR")
    ## change state_sim to 51 based on number of parameters I passed in the Load.py (ln# 7 - 13)  from the Fold0_Train2.csv file.
    #parser.add_argument('--state_dim', type=int, default=38)
    #parser.add_argument('--state_dim', type=int, default=51)
    parser.add_argument('--state_dim', type=int, default=136)  ##updated March 2025
    parser.add_argument('--action_dim', type=int, default=2401)
    parser.add_argument("--h_dim", type=int, default=60)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--drop_p", type=float, default=0.5)
    parser.add_argument("--missing_rate", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate") ### start with 0.001 or 10^-4 generally 3e-4 and then adjust based on the results.
    parser.add_argument("--lr_decay", type=float, default=0.99, help="Learning rate decay")
    parser.add_argument("--lr_step", type=int, default=2, help="Learning rate decay stepsize")
    parser.add_argument("--w_decay", default=0.0001, type=float, help="Weight decay (lambda) ℓ2 regularization")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    parser.add_argument("--total_epoch", type=int, default=3000, help="# of epochs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU number")
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--lam_reg", type=float, default=1.0, help="Regularization loss coefficient")
    parser.add_argument("--lam_critic", type=float, default=1.0, help="Critic loss coefficient")
    parser.add_argument("--lam_actor", type=float, default=1.0, help="Actor loss coefficient")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001, help="Tau coefficient - soft target update")
    parser.add_argument("--n_tokens", type=int, default=20, help="n_tokens")
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--fold", type=int, default=0, help="5-fold cross validation")
    parser.add_argument("--max_timestep", default=72, type=int)
    parser.add_argument(
        "--categorical_variables",
        type=lambda s: s.split(','),
        default="race,admission_type,admission_location,discharge_location,insurance,language,marital_status,event",
        help="Comma-separated list of categorical variables (e.g. 'var1,var2,var3')."
    )
    parser.add_argument("--config_path", type=str, default="./Configuration/")
    ## adding data-path as location of the files.
    parser.add_argument("--data_path", type=str, default="./Data/")
    ## adding the 3 files separately instead of within data_path
    parser.add_argument("--train-data", type=str, default="Fold0_Train3.csv")
    parser.add_argument("--val-data", type=str, default="Fold0_Val3.csv")
    parser.add_argument("--test-data", type=str, default="Fold0_Test3.csv")
    parser.add_argument("--results_path", type=str, default="./Results/")

    return parser.parse_args()

if "__main__" == __name__:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    exp_name = f"ADTR_Fold{args.fold}"

    # Save configuration
    ut.save_configuration(args.config_path, exp_name+"_Configuration.txt", args)
    tr.run(args, device, exp_name)
