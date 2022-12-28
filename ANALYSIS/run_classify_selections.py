from timeit import default_timer
import os
import shared_data as sd
from datetime import timedelta


start_time = default_timer()

for sel_name in sd.SELECTION_DICT:

    for model_name in sd.PARAMS:

        os.system(f"python3 classify_selection.py {sel_name} {model_name};")

total_time = default_timer() - start_time
print(f"Total time {timedelta(seconds=total_time)}")
