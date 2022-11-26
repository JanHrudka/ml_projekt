from datetime import timedelta
import MDAnalysis as mda
import ml_models as mm
import numpy as np
import shared_data as sd
import utility_functions as uf
import sys
import timeit

if __name__ == "__main__":

    start_time = timeit.default_timer()

    sel_name = sys.argv[1]
    model_name = sys.argv[2]

    y_train = sd.y
    y_test = sd.y

    y_model_array = np.zeros((len(sd.trj_combinations), len(y_test)))

    for comb_ind, train_combination in enumerate(sd.trj_combinations):

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

        models = mm.MLModels(X_train, X_test, y_train, y_test)
        model = getattr(models, model_name)
        y_model = model(sd.MODEL_KWARGS_DICT[model_name])

        y_model_array[comb_ind, :] = y_model.reshape(len(y_test))

    np.save((f"{sd.RESULTS_DIR}/y_pred_{sel_name}_{model_name}.npy"),
            y_model_array)

    total_time = timeit.default_timer() - start_time
    print(f"Time {timedelta(seconds=total_time)} {sel_name} {model_name}")
