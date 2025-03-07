import pandas as pd
import torch
import numpy as np
import Utils as ut
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


""" Indices of the interested EHR variables """
## For PJ Vent subset data Fold0_Train3.csv 1, 5-7, 9-11, 131-132, 141-205
demo_idx = np.array([1,5,6,7,9,10,11, 131, 132, 141 , 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205])
## 13, 15, 18-47
vital_idx = np.array([13, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47])
## 48-71
lab_idx = np.array([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71])
sofa_idx = np.array([103, 104, 105])
action_idx = np.array([140])

## adding sequence_idx = index for col 'hr'
sequence_idx = np.array([12])

## adding index for mortality and reward
mortality_idx = np.array([8])
reward_idx = np.array([127])


## for dummy data Fold0_Train2.csv
# demo_idx = np.array([4,5,6,7,8,58,59,60])
# ## 'obs_6' to 'obs_26'
# vital_idx = np.array([9,10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29])
# lab_idx = np.array([30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])
# sofa_idx = np.array([46,47,48,49,50,51,52,53,54,55,56,57])
# action_idx = np.array([2])


## updating to use the load_fold_new(args) function - PJ Mar 5th 2025
# def load_fold(args):
#
#     trainset = CustomDataset(args.data_path, interval=args.interval, fold=args.fold, mode="Train", missing_rate=args.missing_rate, max_length=args.max_timestep, n_class=25)
#     validset = CustomDataset(args.data_path, interval=args.interval, fold=args.fold, mode="Valid", missing_rate=args.missing_rate, max_length=args.max_timestep, n_class=25)
#     testset = CustomDataset(args.data_path, interval=args.interval, fold=args.fold, mode="Test", missing_rate=args.missing_rate, max_length=args.max_timestep, n_class=25)
#
#     trainloader = DataLoader(trainset, args.bs, shuffle=True)
#     validloader = DataLoader(validset, args.bs, shuffle=False)
#     testloader = DataLoader(testset, args.bs, shuffle=False)
#
#     return trainloader, validloader, testloader



def load_fold_new(args):

    ### process non-numeric data in the dataset - this should be refactored into the imputation code later on. Ensure the returned DF order is retained: Train, test then val.
    train_data2, test_data2, val_data2 = ut.process_data_further(args)



    ## Create the CustomDataset
    trainset = CustomDataset(
        # data_path=args.data_path,
        # data=args.data["train"],
        data = train_data2,
        interval=args.interval,
        fold=args.fold,
        mode="Train",
        max_length=args.max_timestep,
    )
    validset = CustomDataset(
        # data_path=args.data_path,
        # data=args.data["val"],
        data = val_data2,
        interval=args.interval,
        fold=args.fold,
        mode="Valid",
        max_length=args.max_timestep,
    )
    testset = CustomDataset(
        # data_path=args.data_path,
        # data=args.data["test"],
        data = test_data2,
        interval=args.interval,
        fold=args.fold,
        mode="Test",
        max_length=args.max_timestep,
    )

    trainloader = DataLoader(trainset, args.bs, shuffle=True)
    validloader = DataLoader(validset, args.bs, shuffle=False)
    testloader = DataLoader(testset, args.bs, shuffle=False)

    return trainloader, validloader, testloader



def scale_dataframe(df, columns):
    """
    Scales the specified columns of the dataframe using StandardScaler.
    Returns the scaled dataframe and the fitted scaler.
    """
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def scale_features(df, indices):
    # Get column names based on the provided indices
    feature_columns = df.columns[indices].tolist()
    scaler = StandardScaler()
    # Fit and transform only the selected columns
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler


class CustomDataset(Dataset):
    ## updated the __init to only send in the dataframe and not the path.
    #def __init__(self, data_path, interval=4, fold=1, mode="Train", missing_rate=0, max_length=20, n_class=25):
    def __init__(self, data, interval=4, fold=1, mode="Train", missing_rate=0, max_length=20, n_class=25):
        self.missing_rate = missing_rate
        self.interval = interval
        self.fold = fold
        self.mode = mode
        self.T = max_length
        self.n_class = n_class

        # if self.mode in ["Train", "train"]:
        #     self.fpath = data_path + f"Fold{self.fold}_Train3.csv"
        # elif self.mode in ["Valid", "valid"]:
        #     self.fpath = data_path + f"Fold{self.fold}_Valid2.csv"
        # elif self.mode in ["Test", "test"]:
        #     self.fpath = data_path + f"Fold{self.fold}_Test2.csv"
        # else:
        #     raise KeyError("Unknown mode. You should select one among train, valid, and test.")
        #
        # self.df = pd.read_csv(self.fpath)

        self.df = data
        self.head = self.df.columns

        # # Identify the columns to scale (for example, all vital signs and lab values)
        # feature_columns = list(self.df.columns[vital_idx, lab_idx, sofa_idx, mortality_idx])  # update indices as needed

        # Optionally, if you want to scale your features:
        # Combine all the indices into one array. If they are numpy arrays:
        all_indices = np.concatenate((vital_idx, lab_idx, sofa_idx, mortality_idx, reward_idx))

        # Apply scaling to those columns:
        # self.df, self.scaler = scale_dataframe(self.df, feature_columns)
        self.df, self.scaler = scale_features(self.df, all_indices)


        #self.pindices = self.df[['subject_id', 'hadm_id', 'stay_id', 'ventnum']].unique()
        ## create my own "r
        self.df["traj"] = (
                self.df["subject_id"].astype(str) + "_" +
                self.df["hadm_id"].astype(str) + "_" +
                self.df["stay_id"].astype(str) + "_" +
                self.df["ventnum"].astype(str)
        )

        self.pindices = self.df["traj"].unique()

    def parse_delta(self, mask, direction, time):
        """
        :param mask: masking vectors
        :param direction: forward or backward
        :param time: Real time interval ("s" in BRITS paper)
        :return:
        """

        if direction == "backward":
            mask = mask[::-1]

        [T, D] = mask.shape
        deltas = []
        for t in range(T):
            if t == 0:
                deltas.append(np.zeros(D))
            else: # t>0
                deltas.append(np.ones(D) * time[t] - np.ones(D) * time[t-1] + (1-mask[t-1]) * deltas[-1])

        return np.array(deltas)

    def calculate_rtg(self, rewards):
        """
        :param rewards: ndarray
        :return:
        """
        return np.flip(np.cumsum(np.flip(rewards)))

    def get_data(self, idx):

        #print("index:"+str(idx))

        condition = self.df.traj == idx
        vitals = self.df[condition].iloc[:, vital_idx].values #[T, 8]
        labs = self.df[condition].iloc[:, lab_idx].values #[T, 22]
        sofas = self.df[condition].iloc[:, sofa_idx].values #[T, 7]
        actions = self.df[condition].iloc[:, action_idx[0]].values #[T,]
        #sequence = self.df[condition].iloc[:, 1].values #[T,]
        sequence = self.df[condition].iloc[:, sequence_idx[0]].values  # [T,]
        #mortality = self.df[condition].iloc[:, -1].values
        mortality = self.df[condition].iloc[:, mortality_idx[0]].values
        #reward = self.df[condition].iloc[:, -1].values * 15
        reward = self.df[condition].iloc[:, reward_idx[0]].values
        demo = self.df[condition].iloc[:, demo_idx].values

        mortality_last = np.array([mortality[-1]])

        # Get SOFA scores  sofa_24hours	 gcs_average	 RASS_AVG_tw_score
        # sofa_res = sofas[:, 0]
        # sofa_coa = sofas[:, 1]
        # sofa_liv = sofas[:, 2]
        # sofa_car = sofas[:, 3]
        # sofa_cns = sofas[:, 4]
        # sofa_ren = sofas[:, 5]
        # sofa_all = sofas[:, -1]

        #print(sofas.shape)

        sofa_res = sofas[:, 1]   ###  GCS average score
        sofa_coa = sofas[:, 2]   ### RASS_AVG_tw_score
        sofa_liv = sofas[:, 0]   ### SOFA 24 hrs
        sofa_car = sofas[:, 1]   ###  GCS average score
        sofa_cns = sofas[:, 2]   ### RASS_AVG_tw_score
        sofa_ren = sofas[:, 0]   ### SOFA 24 hrs
        sofa_all = sofas[:, 0]  ### SOFA 24 hrs

        mask_sofa_res = ~np.isnan(sofa_res)
        mask_sofa_coa = ~np.isnan(sofa_coa)
        mask_sofa_liv = ~np.isnan(sofa_liv)
        mask_sofa_car = ~np.isnan(sofa_car)
        mask_sofa_cns = ~np.isnan(sofa_cns)
        mask_sofa_ren = ~np.isnan(sofa_ren)
        mask_sofa_all = ~np.isnan(sofas[:,:-1])


        # Get EHR variables
        data = np.concatenate((vitals, labs), axis=-1)  # [sequence, variables]
        ###
        mask_data_real = ~ np.isnan(data)  # 1: observed, 0: missing
        delta = self.parse_delta(mask_data_real, direction="forward", time=sequence)
        delta_sofa = self.parse_delta(mask_sofa_all, direction="forward", time=sequence)
        seq_mask = np.ones(shape=(sequence.shape[0]))

        # Pad with 0 in data to construct a mini-batch
        if sequence.shape[0] < self.T:
            #rtg = self.calculate_rtg(np.delete(sofa_all, 0) - np.delete(sofa_all, -1))
            #rtg = np.concatenate((rtg, mortality_last*15))
            rtg = np.flip(np.cumsum(np.flip(reward)))

            demo = np.concatenate((demo, np.zeros((int(self.T - sequence.shape[0]), demo.shape[1]))), axis=0)  # [T, n_variable]
            data = np.concatenate((data, np.zeros((int(self.T - sequence.shape[0]), data.shape[1]))),
                                  axis=0)  # [T, n_variable]
            mask_data_real = np.concatenate((mask_data_real, np.zeros((int(self.T - sequence.shape[0]), mask_data_real.shape[1]))), axis=0)  # [T, n_variable]
            mask_sofa_all = np.concatenate((mask_sofa_all, np.zeros((int(self.T - sequence.shape[0]), mask_sofa_all.shape[1]))), axis=0)

            actions = np.concatenate((actions, np.zeros((int(self.T - sequence.shape[0])))), axis=0)

            mortality = np.concatenate((mortality, np.zeros((int(self.T - sequence.shape[0])))), axis=0)
            # pad_len = int(self.T - sequence.shape[0])
            # # Ensure the zeros array has the same number of dimensions as 'mortality'
            # mortality = np.concatenate((mortality, np.zeros((pad_len, mortality.shape[1]))), axis=0)

            reward = np.concatenate((reward, np.zeros((int(self.T - sequence.shape[0])))), axis=0)
            #reward = np.concatenate((reward, np.zeros((pad_len, reward.shape[1]))), axis=0)

            mask_sofa_res = np.concatenate((mask_sofa_res, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_coa = np.concatenate((mask_sofa_coa, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_liv = np.concatenate((mask_sofa_liv, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_car = np.concatenate((mask_sofa_car, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_cns = np.concatenate((mask_sofa_cns, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            mask_sofa_ren = np.concatenate((mask_sofa_ren, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]

            sofa_res = np.concatenate((sofa_res, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_coa = np.concatenate((sofa_coa, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_liv = np.concatenate((sofa_liv, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_car = np.concatenate((sofa_car, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_cns = np.concatenate((sofa_cns, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_ren = np.concatenate((sofa_ren, np.zeros((int(self.T - sequence.shape[0])))), axis=0)  # [T]
            sofa_all = np.concatenate((sofa_all, np.zeros((int(self.T - sequence.shape[0])))), axis=0) #[T]

            seq_mask = np.concatenate((seq_mask, np.zeros((int(self.T - sequence.shape[0])))), axis=0)

            sequence = np.concatenate((sequence, np.nan * np.ones((int(self.T - sequence.shape[0])))), axis=0)
            #sequence = np.concatenate((sequence, np.nan * np.ones((pad_len, sequence.shape[1]))), axis=0)

        else:
            demo = demo[:self.T]
            data = data[:self.T]
            mask_data_real = mask_data_real[:self.T]
            delta_real = delta[:self.T]
            delta_sofa = delta_sofa[:self.T]
            actions = actions[:self.T]
            mask_sofa_res = mask_sofa_res[:self.T]
            mask_sofa_coa = mask_sofa_coa[:self.T]
            mask_sofa_liv = mask_sofa_liv[:self.T]
            mask_sofa_car = mask_sofa_car[:self.T]
            mask_sofa_cns = mask_sofa_cns[:self.T]
            mask_sofa_ren = mask_sofa_ren[:self.T]
            mask_sofa_all = mask_sofa_all[:self.T]

            sofa_all = sofa_all[:self.T]
            sofa_res = sofa_res[:self.T]
            sofa_coa = sofa_coa[:self.T]
            sofa_liv = sofa_liv[:self.T]
            sofa_car = sofa_car[:self.T]
            sofa_cns = sofa_cns[:self.T]
            sofa_ren = sofa_ren[:self.T]
            sequence = sequence[:self.T]
            seq_mask = seq_mask[:self.T]
            mortality = mortality[:self.T]
            reward = reward[:self.T]


        record = dict()
        record["observation"] = torch.from_numpy(np.nan_to_num(data, nan=0)).float()
        record["action"] = torch.from_numpy(actions.astype("long"))
        record["sequence"] = torch.from_numpy(np.nan_to_num(sequence, nan=0)).float()
        record["seq_mask"] = torch.from_numpy((seq_mask.astype("int32")))
        record["mask_data_real"] = torch.from_numpy(mask_data_real.astype("int32"))

        record["mask_sofa_res"] = torch.from_numpy(mask_sofa_res).float()
        record["mask_sofa_coa"] = torch.from_numpy(mask_sofa_coa).float()
        record["mask_sofa_liv"] = torch.from_numpy(mask_sofa_liv).float()
        record["mask_sofa_car"] = torch.from_numpy(mask_sofa_car).float()
        record["mask_sofa_cns"] = torch.from_numpy(mask_sofa_cns).float()
        record["mask_sofa_ren"] = torch.from_numpy(mask_sofa_ren).float()
        record["mask_sofa_all"] = torch.from_numpy(mask_sofa_all).float()

        record["sofa_res"] = torch.from_numpy(np.nan_to_num(sofa_res, nan=0).astype("long"))
        record["sofa_coa"] = torch.from_numpy(np.nan_to_num(sofa_coa, nan=0).astype("long"))
        record["sofa_liv"] = torch.from_numpy(np.nan_to_num(sofa_liv, nan=0).astype("long"))
        record["sofa_car"] = torch.from_numpy(np.nan_to_num(sofa_car, nan=0).astype("long"))
        record["sofa_cns"] = torch.from_numpy(np.nan_to_num(sofa_cns, nan=0).astype("long"))
        record["sofa_ren"] = torch.from_numpy(np.nan_to_num(sofa_ren, nan=0).astype("long"))
        record["sofa_all"] = torch.from_numpy(np.nan_to_num(sofa_all, nan=0).astype("long"))

        record["mortality"] = torch.from_numpy(mortality).float()
        record["reward"] = torch.from_numpy(reward).float()
        record["demo"] = torch.from_numpy(np.nan_to_num(demo, nan=0)).float()
        record["mortality_last"] = torch.from_numpy(mortality_last).float()


        return record

    def __getitem__(self, idx):
        return self.get_data(self.pindices[idx])

    def __len__(self):
        return len(self.pindices)  # the number of patients