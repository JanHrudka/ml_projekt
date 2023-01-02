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

SELECTION_DICT = {
    "uL22": "segid S",
    "uL22_tip": "segid S and resid 85:95",
    "uL22_and_surroundings": "((around 10 segid S) and "
                             "not (segid y)) or (segid S)",
    "uL22_tip_and_surroundings": "(segid S and "
                                 "(resid 89:93)) or "
                                 "(segid A and (resid 746 "
                                 "747 748 750 751 1614))"
}

SELECTION_NAMES_PLOTTING = {"uL22": "uL22",
                            "uL22_tip": "uL22 tip",
                            "uL22_and_surroundings": "uL22+surr.",
                            "uL22_tip_and_surroundings": "uL22 tip+surr."
                            }

PARAMS = {"svc": {"C": [0.1, 1, 10]},
          "logistic_regression": {"max_iter": [10000],
                                  "C": [0.1, 1, 10]},
          "decision_tree": {"criterion": ["gini", "entropy", "log_loss"]},
          "random_forest": {"criterion": ["gini", "entropy", "log_loss"],
                            "n_estimators": [10, 50, 100]},
          "knn": {"n_neighbors": [3, 4, 5],
                  "p": [1, 2]},
          "mlp": {"hidden_layer_sizes": [(50,), (100,), (150,)],
                  "alpha": [0.0001, 0.0005, 0.001],
                  "learning_rate": ["adaptive"],
                  "max_iter": [50]},
          "gnb": {"var_smoothing": [1e-8, 1e-9, 1e-10]},
          "ada": {"n_estimators": [10, 50, 100],
                  "algorithm": ['SAMME', 'SAMME.R']
                  }
          }

MODEL_NAMES_PLOTTING = {"svc": "Support Vector Classifier",
                        "logistic_regression": "Logistic Regression",
                        "decision_tree": "Decision Tree",
                        "random_forest": "Random Forest",
                        "knn": "K-Nearest Neighbors",
                        "mlp": "Multilayer Perceptron",
                        "gnb": "Gaussian Naive Bayes",
                        "ada": "AdaBoost Classifier"
                        }

# all possible combinations of trajectory pairs
trj_combinations = []

for i in range(N_TRJ):

    for j in range(i+1, N_TRJ):

        trj_combinations.append([i, j])

TOTAL_TRJ_LEN = (TRJ_LEN * N_TRJ)

y = np.zeros(TOTAL_TRJ_LEN)
y[:TOTAL_TRJ_LEN//2] = 1
