import numpy as np


def extract_trj(aligned_u, selection_keyword="all"):
    """
    Extract the trajectory from the universe object into a numpy array
    and calculate the mean structure of this trajectory.

    Parameters:

        aligned_u (object): Universe object created from the
        aligned trajectories.

    Returns:

        X (array (n_frames, 3 * n_atoms)): Trajectory extracted into
        an array.
    """

    trj = aligned_u.trajectory
    atomgroup = aligned_u.select_atoms(selection_keyword)

    n_frames = len(trj)
    n_atoms = len(atomgroup)

    X = np.zeros((n_frames, 3 * n_atoms))
    for frame_index, _ in enumerate(trj):

        X[frame_index, :] = atomgroup.positions.ravel()

    return X
