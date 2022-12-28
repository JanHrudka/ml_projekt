from datetime import timedelta
from sklearn.model_selection import GridSearchCV
import json
import MDAnalysis as mda
import ml_models as mm
import numpy as np
import os
import shared_data as sd
from sklearn.preprocessing import StandardScaler
import utility_functions as uf
import sys
import timeit


def get_X_for_combination(sel_name, train_combination):

    test_combination = np.setdiff1d(sd.TRJ_INDICES, train_combination)
    train_A, train_B = train_combination
    test_A, test_B = test_combination

    uni_train = mda.Universe(
        f"{sd.PREPRO_DATA_DIR}/{sel_name}_combined.pdb",
        f"{sd.PREPRO_DATA_DIR}/{sel_name}_combined_{train_A}{train_B}.xtc"
    )
    uni_test = mda.Universe(
        f"{sd.PREPRO_DATA_DIR}/{sel_name}_combined.pdb",
        f"{sd.PREPRO_DATA_DIR}/{sel_name}_combined_{test_A}{test_B}.xtc"
    )

    X_train = uf.extract_trj(uni_train)
    X_test = uf.extract_trj(uni_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, X_test


if __name__ == "__main__":

    start_time = timeit.default_timer()

    sel_name = sys.argv[1]
    model_name = sys.argv[2]

    y_train = sd.y
    y_test = sd.y

    models = mm.MLModels()
    model = getattr(models, model_name)

    # HYPERPARAMETER TUNING
    # perform grid search hyperparameter tuning on one train/test split
    # combination for the small selection uL22_tip
    if os.path.exists(f"{sd.RESULTS_DIR}/best_params_{model_name}.json"):

        with open(
            f"{sd.RESULTS_DIR}/best_params_{model_name}.json", "r"
        ) as param_file:

            best_params = json.load(param_file)

    else:

        X_train, X_test = get_X_for_combination("uL22_tip",
                                                sd.trj_combinations[0])

        grid_search = GridSearchCV(
            estimator=model(),
            param_grid=sd.PARAMS[model_name],
            scoring="accuracy"
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_

        with open(
            f"{sd.RESULTS_DIR}/best_params_{model_name}.json", "w"
        ) as param_file:

            json.dump(best_params, param_file)

    # K-FOLD CROSSVALIDATION
    # best hyperparameters from grid search are then used for k-fold
    # crossvalidation
    y_pred_array = np.zeros((len(sd.trj_combinations), len(y_test)))

    for comb_ind, train_combination in enumerate(sd.trj_combinations):

        X_train, X_test = get_X_for_combination(sel_name, train_combination)

        classifier = model(best_params)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        y_pred_array[comb_ind, :] = y_pred.reshape(len(y_test))

    np.save((f"{sd.RESULTS_DIR}/y_pred_{sel_name}_{model_name}.npy"),
            y_pred_array)

    total_time = timeit.default_timer() - start_time
    print(f"Time {timedelta(seconds=total_time)} {sel_name} {model_name}")
