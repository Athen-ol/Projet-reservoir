import csv
import numpy as np

def read_period_csv(csv_path): #converts csv to lists
    PERIOD = []
    TIMEDELTA = []
    DST = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            PERIOD.append(row["period"])
            TIMEDELTA.append(row["timedelta"])
            DST.append(row["dst"])

    return PERIOD, TIMEDELTA, DST


def transform_dst_to_int(dst_list): #converts list of stings into list of int
    return [int(float(dst)) for dst in dst_list]


def convert_timedelta_to_int(timedelta_sting): #takes the string format "X days HH:MM:SS" and converts it to int hours
    days_part, time_part = timedelta_sting.split(" days ")   
    DAY = int(days_part)
    HOUR = int(time_part.split(":")[0])
    return DAY * 24 + HOUR

def transform_timedelta_to_int(timedelta_list): #converts list of stings into list of int
    return [convert_timedelta_to_int(timedelta) for timedelta in timedelta_list]

def normalize_dst(dst_list): #normalize dst between -1 and 1
    min_dst = min(dst_list)
    max_dst = max(dst_list)
    use_to_normalize = max(abs(min_dst), abs(max_dst))
    return [(dst) / (use_to_normalize) for dst in dst_list]


def split_datasets(period_list, timedelta_list, dst_list): #splits the data into 3 sets according to sample period
    SET_A = [[],[],[]]
    SET_B = [[],[],[]]
    SET_C = [[],[],[]]
    for i in range(len(period_list)):
        if period_list[i] == 'train_a':
            SET_A[0].append(period_list[i])
            SET_A[1].append(timedelta_list[i])
            SET_A[2].append(dst_list[i])
        elif period_list[i] == 'train_b':
            SET_B[0].append(period_list[i])
            SET_B[1].append(timedelta_list[i])
            SET_B[2].append(dst_list[i])
        elif period_list[i] == 'train_c':
            SET_C[0].append(period_list[i])
            SET_C[1].append(timedelta_list[i])
            SET_C[2].append(dst_list[i])
    return SET_A, SET_B, SET_C


def init_dataset(csv_path):
    PERIOD, TIMEDELTA, DST = read_period_csv(csv_path)
    SET_A, SET_B, SET_C = split_datasets(PERIOD, transform_timedelta_to_int(TIMEDELTA), normalize_dst(transform_dst_to_int(DST)))
    SET_A = np.array(SET_A[2]).reshape(-1, 1)
    SET_B = np.array(SET_B[2]).reshape(-1, 1)
    SET_C = np.array(SET_C[2]).reshape(-1, 1)
    return SET_A, SET_B, SET_C

if __name__ == "__main__":
    csv_path = "dst_labels.csv"
    SET_A, SET_B, SET_C = init_dataset(csv_path)