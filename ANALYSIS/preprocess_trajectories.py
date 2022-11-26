import MDAnalysis as mda
import shared_data as sd

PDB_LIST = [f"T{i}/ribo.pdb" for i in range(sd.N_TRJ)]
XTC_LIST = [f"T{i}/ribo.xtc" for i in range(sd.N_TRJ)]
SLICED_XTC_LIST = [f"T{i}/sliced_trj.xtc" for i in range(sd.N_TRJ)]

COMMON_ATOMS = "not segid y"

system_trj_list = []

for system_name, system_path in sd.SYSTEMS.items():

    trj_list = []

    for pdb, xtc, sliced_xtc in zip(PDB_LIST, XTC_LIST,
                                    SLICED_XTC_LIST):

        single_trj_uni = mda.Universe(f"{system_path}/{pdb}",
                                      f"{system_path}/{xtc}")
        single_trj_sel = single_trj_uni.select_atoms("all")
        single_trj_sel.write(
            f"{system_path}/{sliced_xtc}",
            frames=single_trj_uni.trajectory[
                sd.TRJ_START_FRAME:sd.TRJ_END_FRAME
            ]
        )

        trj_list.append(f"{system_path}/{sliced_xtc}")

    uni = mda.Universe(f"{system_path}/{PDB_LIST[0]}", trj_list)
    sel = uni.select_atoms(COMMON_ATOMS)
    sel.write(f"{system_path}/{system_name}.pdb")
    sel.write(f"{system_path}/{system_name}.xtc",
              frames="all")

    system_trj_list.append(f"{system_path}/{system_name}.xtc")

uni_combined = mda.Universe(f"{sd.SYSTEMS['NONE']}/NONE.pdb", system_trj_list)

for sel_name, sel_keyword in sd.SELECTION_DICT.items():

    sel_combined = uni_combined.select_atoms(sel_keyword)
    sel_combined.write(f"{sd.PREPRO_DATA_DIR}/{sel_name}_combined.pdb")

    for trj_combination in sd.trj_combinations:

        trj_A, trj_B = trj_combination
        indices = []

        for i, _ in enumerate(sd.SYSTEMS):

            for trj in trj_combination:

                for index in range(
                    trj * sd.TRJ_LEN +
                    i * sd.N_TRJ * sd.TRJ_LEN,
                    (trj + 1) * sd.TRJ_LEN +
                    i * sd.N_TRJ * sd.TRJ_LEN,
                ):

                    indices.append(index)

        sel_combined.write(
            f"{sd.PREPRO_DATA_DIR}/{sel_name}_combined_{trj_A}{trj_B}.xtc",
            frames=uni_combined.trajectory[indices]
        )
