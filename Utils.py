import os, glob, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, jaccard_score, classification_report
from sklearn.preprocessing import OneHotEncoder


def save_configuration(path, name, arg):
    with open(path + name, "w") as f:
        json.dump(arg.__dict__, f, indent=2)
    print("Save configuration.")

def read_file_name(query, pool_path):
    answer = glob.glob(pool_path + query)
    return answer

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

    return path

def delete_dir(path):
    if os.path.isfile(path):
        os.remove(path)

def calculate_wis(probs_list, trues_list, rewards_list, masks_list, gamma=0.99):
    """ Reference: https://github.com/yinchangchang/DAC/blob/a36cef7b94464d07eca4f317e3c235aca7fcdd81/tools/py_op.py#L130 """

    rho_list = []
    for probs, rewards, masks, reals in zip(probs_list, rewards_list, masks_list, trues_list):
        #print(probs, rewards, masks, reals)

        assert len(probs) == len(rewards) == len(masks) == len(reals)
        rho = []
        for prob, reward, mask, real in zip(probs, rewards, masks, reals):  # T
            if mask == 0:
                break
            prob = min(1, max(0, prob[real]))
            if len(rho) == 0:
                rho.append(prob)
            else:
                rho.append(prob * rho[-1])
        rho_list.append(rho)

    max_step = max([len(rho) for rho in rho_list])

    w_list = []
    for i in range(max_step):
        w_h = []
        for rho in rho_list:
            if len(rho) > i:
                w_h.append(rho[i])

        w_list.append(np.mean(w_h))

    v_list = []
    for rho, rs, ms in zip(rho_list, rewards_list, masks_list):
        h = len(rho)
        if h <= 1:
            continue

        assert rho[h - 1] <= 1
        assert w_list[h - 1] <= 1
        assert len(rs) > 0
        assert rs[h - 1] != 0
        v_wis = rho[h - 1] / (w_list[h - 1] + 1e-6) * rs[h - 1] * np.power(gamma, len(rho) - 1)
        v_list.append(v_wis)

    return np.mean(v_list)


def calculate_metric(reals, preds, masks, probs, rewards):

    a_true = reals.flatten()
    a_pred = preds.flatten()
    mask = masks.flatten()

    mask_idx = np.nonzero(mask)
    a_true = a_true[mask_idx[0]]
    a_pred = a_pred[mask_idx[0]]

    acc = accuracy_score(a_true, a_pred)

    jaccard = jaccard_score(a_true, a_pred, average="micro")

    report = classification_report(a_true, a_pred, output_dict=True)
    recall = report["macro avg"]["recall"]

    wis = calculate_wis(probs, reals, rewards, masks)


    return acc, jaccard, recall, wis




class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    data_path = './',
    interval = 1,
    fold = 0,
    mode = 'Train',
    max_length = 100,
    bs = 32


def process_data_further(args):

    train_df = pd.read_csv(f"{args.data_path}/{args.data['train']}", index_col=None, sep='\t').reset_index(drop=True)
    test_df = pd.read_csv(f"{args.data_path}/{args.data['test']}", index_col=None, sep='\t').reset_index(drop=True)
    val_df = pd.read_csv(f"{args.data_path}/{args.data['val']}", index_col=None, sep='\t').reset_index(drop=True)

    ### Pre-Processing of Continuous Action Features
    # ------------------------------
    # Process continuous action-space features first
    # ------------------------------

    # For each of train, test, val: add height4PBW, calculate PBW, convert tidal volume, and discretize the action features.
    for df in [train_df, test_df, val_df]:
        # Create 'height4PBW' using existing height or imputed average based on gender
        df = add_height4PBW(df)
        # Calculate predicted body weight (PBW)
        df = add_pbw(df)
        # Convert tidal_volume_merged (mL) to tidal_volume_ml_per_kg using PBW
        df = convert_tidal_volume(df)
        # Discretize each of the continuous action-space features
        df = discretize_peep(df)
        df = discretize_fio2(df)
        df = discretize_tidal_volume(df)
        df = discretize_rr(df)
        # Combine the encoded action features into one column 'actions_encoded_combined'
        df = combine_actions(df)

    # ------------------------------
    # Process categorical non-action features using one-hot encoding
    # ------------------------------
    # Define categorical features for one-hot encoding

    # Initialize encoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # categorical_var_list = ['anchor_year_group', 'race', 'first_hosp_stay', 'first_icu_stay', 'admission_type',
    #                               'admit_provider_id','admission_location',	'discharge_location', 'insurance', 'language',
    #                               'marital_status']
    categorical_var_list = args.categorical_variables

    # Fit only on training data
    # train_categorical = train_df[['anchor_year_group', 'race', 'first_hosp_stay', 'first_icu_stay', 'admission_type',
    #                               'admit_provider_id','admission_location',	'discharge_location', 'insurance', 'language',
    #                               'marital_status']]
    train_categorical = train_df[categorical_var_list]
    encoder.fit(train_categorical)

    # Transform train, test, validation datasets
    train_encoded = encoder.transform(train_categorical)
    # test_encoded = encoder.transform(test_df[['anchor_year_group', 'race', 'first_hosp_stay', 'first_icu_stay', 'admission_type',
    #                               'admit_provider_id','admission_location',	'discharge_location', 'insurance', 'language',
    #                               'marital_status']])
    # val_encoded = encoder.transform(val_df[['anchor_year_group', 'race', 'first_hosp_stay', 'first_icu_stay', 'admission_type',
    #                               'admit_provider_id','admission_location',	'discharge_location', 'insurance', 'language',
    #                               'marital_status']])
    test_encoded = encoder.transform(test_df[categorical_var_list])
    val_encoded = encoder.transform(val_df[categorical_var_list])


    # Convert to DataFrame with column names
    # encoded_feature_names = encoder.get_feature_names_out(['anchor_year_group', 'race', 'first_hosp_stay', 'first_icu_stay', 'admission_type',
    #                               'admit_provider_id','admission_location',	'discharge_location', 'insurance', 'language',
    #                               'marital_status'])
    encoded_feature_names = encoder.get_feature_names_out(categorical_var_list)

    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_feature_names)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_feature_names)
    val_encoded_df = pd.DataFrame(val_encoded, columns=encoded_feature_names)

    # Drop original categorical columns and concatenate encoded features
    # train_df = train_df.drop(columns=['anchor_year_group', 'race', 'first_hosp_stay', 'first_icu_stay', 'admission_type',
    #                               'admit_provider_id','admission_location',	'discharge_location', 'insurance', 'language',
    #                               'marital_status']).reset_index(drop=True)
    # test_df = test_df.drop(columns=['anchor_year_group', 'race', 'first_hosp_stay', 'first_icu_stay', 'admission_type',
    #                               'admit_provider_id','admission_location',	'discharge_location', 'insurance', 'language',
    #                               'marital_status']).reset_index(drop=True)
    # val_df = val_df.drop(columns=['anchor_year_group', 'race', 'first_hosp_stay', 'first_icu_stay', 'admission_type',
    #                               'admit_provider_id','admission_location',	'discharge_location', 'insurance', 'language',
    #                               'marital_status']).reset_index(drop=True)

    train_df = train_df.drop(columns=categorical_var_list).reset_index(drop=True)
    test_df = test_df.drop(columns=categorical_var_list).reset_index(drop=True)
    val_df = val_df.drop(columns=categorical_var_list).reset_index(drop=True)

    ### do a "join"
    train_data = pd.concat([train_df, train_encoded_df], axis=1)
    test_data = pd.concat([test_df, test_encoded_df], axis=1)
    val_data = pd.concat([val_df, val_encoded_df], axis=1)



    print("encoded non-numeric data")
    print("train_data: ", train_data.columns.to_list())
    print("test_data: ", test_data.columns.to_list())
    print("val_data: ", val_data.columns.to_list())

    ## make temp directory to store intermediate files and name the directory after the experiment name.
    temp_dir = create_dir(f"{args.data_path}/temp_{args.exp_name}")
    print("temp dir: ", temp_dir)

    ## save train data, test data and val data to the temp directory.
    # train_data.to_csv(f"{args.data_path}/train_data_OHencoded_terminal.csv", index=False)
    # test_data.to_csv(f"{args.data_path}/test_data_OHencoded_terminal.csv", index=False)
    # val_data.to_csv(f"{args.data_path}/val_data_OHencoded_terminal.csv", index=False)
    train_data.to_csv(f"{temp_dir}/train_data_OHencoded_terminal.csv", index=False)
    test_data.to_csv(f"{temp_dir}/test_data_OHencoded_terminal.csv", index=False)
    val_data.to_csv(f"{temp_dir}/val_data_OHencoded_terminal.csv", index=False)

    return train_data, test_data, val_data


# Function to impute height and create "height4PBW"
def impute_height(row):
    if pd.notnull(row['height']):
        return row['height']
    else:
        if row['gender'].strip().upper() == 'M':
            return 70.0  # average male height in inches
        elif row['gender'].strip().upper() == 'F':
            return 65.0  # average female height in inches
        else:
            return np.nan


def add_height4PBW(df):
    df['height4PBW'] = df.apply(impute_height, axis=1)
    return df


# Function to calculate Predicted Body Weight (PBW)
def calculate_pbw(row):
    """
    Calculate predicted body weight (PBW) using height and one-hot encoded gender columns.

    For one-hot encoded columns:
      - If 'gender_M' equals 1, then male: PBW (kg) = 50 + 2.3*(height4PBW - 60)
      - If 'gender_F' equals 1, then female: PBW (kg) = 45.5 + 2.3*(height4PBW - 60)

    If one-hot columns are not present, fall back to using the 'gender' column.
    """
    if pd.isnull(row['height4PBW']):
        return np.nan

    # Check if one-hot encoded gender columns exist in the row
    if ('gender_M' in row) and ('gender_F' in row):
        if row['gender_M'] == 1:
            return 50 + 2.3 * (row['height4PBW'] - 60)
        elif row['gender_F'] == 1:
            return 45.5 + 2.3 * (row['height4PBW'] - 60)
        else:
            return np.nan
    else:
        # Fallback to using the original "gender" column if one-hot columns are not available.
        gender = str(row.get('gender', '')).strip().upper()
        if gender == 'M':
            return 50 + 2.3 * (row['height4PBW'] - 60)
        elif gender == 'F':
            return 45.5 + 2.3 * (row['height4PBW'] - 60)
        else:
            return np.nan


def add_pbw(df):
    df['PBW'] = df.apply(calculate_pbw, axis=1)
    return df


# Function to convert tidal volume to ml/kg
def convert_tidal_volume(df):
    df['tidal_volume_ml_per_kg'] = df['tidal_volume_merged'] / df['PBW']
    return df


# Functions to discretize each variable using pd.cut
def discretize_peep(df):
    bins_peep = [-0.001, 4, 7, 10, 13, 16, 19, np.inf]
    labels_peep = ['PEEP1', 'PEEP2', 'PEEP3', 'PEEP4', 'PEEP5', 'PEEP6', 'PEEP7']
    df['peep_encoded'] = pd.cut(df['peep'], bins=bins_peep, labels=labels_peep, right=True)
    return df


def discretize_fio2(df):
    bins_fio2 = [20, 40, 50, 60, 70, 80, 90, np.inf]
    labels_fio2 = ['Fio21', 'Fio22', 'Fio23', 'Fio24', 'Fio25', 'Fio26', 'Fio27']
    df['fio2_merged_encoded'] = pd.cut(df['fio2_merged'], bins=bins_fio2, labels=labels_fio2, right=True)
    return df


def discretize_tidal_volume(df):
    bins_vt = [0, 2, 4, 6, 8, 10, 12, np.inf]
    labels_vt = ['Vt1', 'Vt2', 'Vt3', 'Vt4', 'Vt5', 'Vt6', 'Vt7']
    df['tidal_volume_encoded'] = pd.cut(df['tidal_volume_ml_per_kg'], bins=bins_vt, labels=labels_vt, right=True)
    return df


def discretize_rr(df):
    bins_rr = [0, 12, 16, 20, 24, 28, 32, np.inf]
    labels_rr = ['RR1', 'RR2', 'RR3', 'RR4', 'RR5', 'RR6', 'RR7']
    df['respiratory_rate_encoded'] = pd.cut(df['respiratory_rate_merged'], bins=bins_rr, labels=labels_rr, right=True)
    return df


# Function to combine the encoded columns into one string
def combine_actions_old(df):
    def combine(row):
        peep_code = row['peep_encoded'] if pd.notnull(row['peep_encoded']) else 'NA'
        fio2_code = row['fio2_merged_encoded'] if pd.notnull(row['fio2_merged_encoded']) else 'NA'
        vt_code = row['tidal_volume_encoded'] if pd.notnull(row['tidal_volume_encoded']) else 'NA'
        rr_code = row['respiratory_rate_encoded'] if pd.notnull(row['respiratory_rate_encoded']) else 'NA'
        return f"{peep_code}_{fio2_code}_{vt_code}_{rr_code}"

    df['actions_encoded_combined'] = df.apply(combine, axis=1)
    return df


# Function to combine the encoded columns into one integer
def combine_actions(df):
    """
    Compute a unique numeric score for each combination of discretized actions,
    using a base-7 encoding for each of the four features. The mapping is:
      - PEEP: 'PEEP1'->0, 'PEEP2'->1, ..., 'PEEP7'->6
      - FiO₂: 'Fio21'->0, 'Fio22'->1, ..., 'Fio27'->6
      - Tidal Volume: 'Vt1'->0, 'Vt2'->1, ..., 'Vt7'->6
      - Respiratory Rate: 'RR1'->0, 'RR2'->1, ..., 'RR7'->6
    The final score is computed as:
      score = (peep_index * 7^3) + (fio2_index * 7^2) + (vt_index * 7^1) + (rr_index * 7^0) + 1
    This ensures that each unique combination gets a unique integer in [1, 2401].
    """
    # Mapping dictionaries for each encoded feature
    peep_map = {'PEEP1': 0, 'PEEP2': 1, 'PEEP3': 2, 'PEEP4': 3, 'PEEP5': 4, 'PEEP6': 5, 'PEEP7': 6}
    fio2_map = {'Fio21': 0, 'Fio22': 1, 'Fio23': 2, 'Fio24': 3, 'Fio25': 4, 'Fio26': 5, 'Fio27': 6}
    vt_map = {'Vt1': 0, 'Vt2': 1, 'Vt3': 2, 'Vt4': 3, 'Vt5': 4, 'Vt6': 5, 'Vt7': 6}
    rr_map = {'RR1': 0, 'RR2': 1, 'RR3': 2, 'RR4': 3, 'RR5': 4, 'RR6': 5, 'RR7': 6}

    def compute_action_score(row):
        # Retrieve the integer code for each encoded feature; default to 0 if missing
        p = peep_map.get(row.get('peep_encoded', None), 0)
        f = fio2_map.get(row.get('fio2_merged_encoded', None), 0)
        v = vt_map.get(row.get('tidal_volume_encoded', None), 0)
        r = rr_map.get(row.get('respiratory_rate_encoded', None), 0)
        # Combine using base-7 representation:
        score = p * (7**3) + f * (7**2) + v * (7**1) + r * (7**0)
        # Add 1 so that the scores range from 1 to 2401
        return score + 1

    df['actions_encoded_combined'] = df.apply(compute_action_score, axis=1)
    return df

