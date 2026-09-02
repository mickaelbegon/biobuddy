from dataclasses import dataclass, field

import numpy as np


@dataclass
class Kinematics:
    """
    Generalized-coordinate samples associated with a biomechanical model.

    The :meth:`from_bvh` and :meth:`from_fbx` constructors extract kinematics
    independently from model parsing. Rotational coordinates are expressed in
    radians and sample times in seconds.

    Parameters
    ----------
    q
        The generalized coordinates with shape ``(nb_q, nb_frames)``.
    time
        The sample times in seconds with shape ``(nb_frames,)``.
    dof_names
        The DoF names associated with the rows of ``q``.
    """

    q: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    time: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    dof_names: list[str] = field(default_factory=list)

    @classmethod
    def from_bvh(cls, filepath: str) -> "Kinematics":
        """
        Extract generalized-coordinate samples from a BVH file.

        Parameters
        ----------
        filepath
            The path to the BVH file to parse.
        """
        from ..model_parser.bvh import BvhModelParser

        return BvhModelParser(filepath=filepath).to_kinematics()

    @classmethod
    def from_fbx(cls, filepath: str) -> "Kinematics":
        """
        Extract generalized-coordinate samples from an FBX file.

        Parameters
        ----------
        filepath
            The path to the FBX file to parse.
        """
        from ..model_parser.fbx import FbxModelParser

        return FbxModelParser(filepath=filepath).to_kinematics()

    @property
    def frame_count(self) -> int:
        """
        Return the number of kinematic frames.
        """
        return int(self.time.shape[0])
