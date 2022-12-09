import numpy as np

N_TRJ = 4
TRJ_INDICES = [i for i in range(N_TRJ)]
# 1 frame = 0.2 ns
TRJ_END_FRAME = 5000
TRJ_START_FRAME = 1000
TRJ_LEN = TRJ_END_FRAME - TRJ_START_FRAME

PROJECT_DIR = "/home/mcgrathh/SCHOOL/SUP/ml_projekt"
DATA_DIR = f"{PROJECT_DIR}/DATA"
PREPRO_DATA_DIR = f"{PROJECT_DIR}/PREPROCESSED_DATA"
RESULTS_DIR = f"{PROJECT_DIR}/RESULTS"

SYSTEMS = {"NONE": f"{DATA_DIR}/NONE",
           "WHOLE": f"{DATA_DIR}/WHOLE"}

SELECTION_DICT = {"uL22": "segid S",
                  "uL22_tip": "segid S and resid 85:95",
                  "uL22_and_surroundings": "((around 10 segid S) and "
                                           "not (segid y)) or (segid S)",
                  "uL22_tip_and_surroundings": "(segid S and "
                                               "(resid 89:93)) or "
                                               "(segid A and (resid 746 "
                                               "747 748 750 751 1614))"}

MODEL_KWARGS_DICT = {"svc": {"kernel": "rbf"},
                     "logistic_regression": {"max_iter": 10000},
                     "decision_tree": {"n_estimators": 1, "bootstrap": False,
                                       "max_depth": 100},
                     "random_forest": {"n_estimators": 100, "bootstrap": True,
                                       "max_depth": 100}
                     }

# all possible combinations of trajectory pairs
trj_combinations = []

for i in range(N_TRJ):

    for j in range(i+1, N_TRJ):

        trj_combinations.append([i, j])

TOTAL_TRJ_LEN = (TRJ_LEN * N_TRJ)

y = np.zeros(TOTAL_TRJ_LEN)
y[:TOTAL_TRJ_LEN//2] = 1
