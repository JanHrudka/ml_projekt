from datetime import timedelta
import subprocess
import shared_data as sd
from timeit import default_timer


start_time = default_timer()

for sel_name in sd.SELECTION_DICT:

    for model_name in sd.MODEL_KWARGS_DICT:

        process = subprocess.run(
            f"python3 classify_selection.py {sel_name} {model_name}",
            shell=True, check=True
        )

total_time = default_timer() - start_time
print(f"Total time {timedelta(seconds=total_time)}")
