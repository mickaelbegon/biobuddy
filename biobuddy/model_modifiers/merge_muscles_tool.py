from copy import deepcopy
import logging
import numpy as np

from ..components.real.biomechanical_model_real import BiomechanicalModelReal
from ..components.real.force.muscle_real import MuscleReal
from ..components.real.force.via_point_real import ViaPointReal
from ..utils.named_list import NamedList

_logger = logging.getLogger(__name__)


class MuscleMerge:
    def __init__(self, name: str, muscle_names: list[str] | str):
        """
        Initialize a muscle merge configuration.
        This is used to merge muscles together in the model.

        Parameters
        ----------
        name
            The name of the new muscle that will be created by merging the two muscles.
        muscle_names
            The name of the muscles to merge
        """
        if isinstance(muscle_names, str):
            muscle_names = [muscle_names]
        if not isinstance(muscle_names, list):
            raise ValueError(
                f"add_muscles expects a list of the names of the muscles to merge together, got {muscle_names}."
            )

        self.name = name
        self.muscle_names = muscle_names


class MergeMusclesTool:
    def __init__(
        self,
        original_model: BiomechanicalModelReal,
    ):
        """
        Initialize the segment merger tool.

        Parameters
        ----------
        original_model
            The original model to modify by merging segments together.
        """

        # Original attributes
        self.original_model = original_model

        # Extended attributes to be filled
        self.merged_model = BiomechanicalModelReal()
        self.muscles_to_merge = NamedList[MuscleMerge]()

    def add(self, merge_muscle: MuscleMerge):
        self.muscles_to_merge._append(merge_muscle)

    def merge(
        self,
    ) -> BiomechanicalModelReal:
        """
        Merge the model's muscles using the configuration defined in the MergeMusclesTool.
        """
        # Copy the original model
        self.merged_model = deepcopy(self.original_model)

        # Then, modify it based on the merge tasks
        for merge_task in self.muscles_to_merge:

            first_muscle = True
            muscle_group_name = ""
            origin_position = []
            insertion_position = []
            optimal_length = []
            maximal_force = []
            tendon_slack_length = []
            pennation_angle = []
            maximal_velocity = []
            maximal_excitation = []
            via_points = {}

            for muscle_name in merge_task.muscle_names:

                if muscle_name not in self.merged_model.muscle_names:
                    raise ValueError(f"Muscle {muscle_name} not found in the merged model.")

                if first_muscle:
                    # First muscle
                    muscle_group_name = self.merged_model.get_muscle_muscle_group_name(muscle_name)
                    muscle_group = self.merged_model.muscle_groups[muscle_group_name]
                    muscle_type = deepcopy(muscle_group.muscles[muscle_name].muscle_type)
                    state_type = deepcopy(muscle_group.muscles[muscle_name].state_type)
                    for via_point in muscle_group.muscles[muscle_name].via_points:
                        if via_point.condition is not None:
                            raise NotImplementedError(
                                f"Conditional via points not implemented for muscle, got a conditional via point in {via_point.name} in the muscle {muscle_name}"
                            )
                        if via_point.movement is not None:
                            raise NotImplementedError(
                                f"Moving via points not implemented for muscle, got a movement via point in {via_point.name} in the muscle {muscle_name}"
                            )
                        via_points[via_point.name] = {
                            "position": [via_point.position],
                            "parent": via_point.parent_name,
                        }
                    first_muscle = False
                else:
                    # Other muscles
                    if (
                        self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].muscle_group
                        != muscle_group_name
                    ):
                        raise ValueError(f"Muscle {muscle_name} does not belong to muscle group {muscle_group_name}")
                    if (
                        self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].muscle_type
                        != muscle_type
                    ):
                        raise ValueError(
                            f"The muscle type of the muscles is not all the same, "
                            f"got {muscle_type} and {self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].muscle_type}"
                        )
                    if self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].state_type != state_type:
                        raise ValueError(
                            f"The muscle state type of the muscles is not all the same, "
                            f"got {state_type} and {self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].state_type}"
                        )
                    if self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].nb_via_points != len(
                        via_points.keys()
                    ):
                        raise ValueError(
                            f"The are not the same number of via points for the muscle {muscle_name} ({self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].nb_via_points}) and"
                            f"for the muscle {merge_task.muscle_names[0]} ({len(via_points.keys())})."
                        )
                    for via_point in self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].via_points:
                        if via_point.condition is not None:
                            raise NotImplementedError(
                                f"Conditional via points not implemented for muscle, got a conditional via point in {via_point.name} in the muscle {muscle_name}"
                            )
                        if via_point.movement is not None:
                            raise NotImplementedError(
                                f"Moving via points not implemented for muscle, got a movement via point in {via_point.name} in the muscle {muscle_name}."
                            )
                        # Find the right via point
                        candidates = []
                        distance = []
                        for via_name in via_points.keys():
                            if via_points[via_name]["parent"] == via_point.parent_name:
                                candidates += [via_name]
                                distance += [
                                    np.linalg.norm(via_points[via_name]["position"][0][:3] - via_point.position[:3])
                                ]
                        if len(candidates) == 0:
                            raise ValueError(
                                f"The via_point {via_point.name} from muscle {muscle_name} could not be "
                                f"matched with any via points of muscle {merge_task.muscle_names[0]} with via points {via_points.keys()}"
                            )
                        via_point_index = np.argmin(np.array(distance))
                        via_points[candidates[via_point_index]]["position"] += [via_point.position]

                origin_position += [
                    self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].origin_position.position
                ]
                insertion_position += [
                    self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].insertion_position.position
                ]
                if self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].optimal_length is not None:
                    optimal_length += [
                        self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].optimal_length
                    ]
                if self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_force is not None:
                    maximal_force += [
                        self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_force
                    ]
                if (
                    self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].tendon_slack_length
                    is not None
                ):
                    tendon_slack_length += [
                        self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].tendon_slack_length
                    ]
                if self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].pennation_angle is not None:
                    pennation_angle += [
                        self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].pennation_angle
                    ]
                if self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_velocity is not None:
                    maximal_velocity += [
                        self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_velocity
                    ]
                if (
                    self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_excitation
                    is not None
                ):
                    maximal_excitation += [
                        self.merged_model.muscle_groups[muscle_group_name].muscles[muscle_name].maximal_excitation
                    ]

            # Add the new merged muscle
            origin_position_value = np.mean(np.array(origin_position), axis=0)
            insertion_position_value = np.mean(np.array(insertion_position), axis=0)
            optimal_length_value = float(np.mean(optimal_length)) if optimal_length else None
            maximal_force_value = float(np.sum(maximal_force)) if maximal_force else None
            tendon_slack_length_value = float(np.mean(tendon_slack_length)) if tendon_slack_length else None
            pennation_angle_value = float(np.mean(pennation_angle)) if pennation_angle else None
            maximal_velocity_value = float(np.mean(maximal_velocity)) if maximal_velocity else None
            maximal_excitation_value = float(np.mean(maximal_excitation)) if maximal_excitation else None
            self.merged_model.muscle_groups[muscle_group_name].add_muscle(
                MuscleReal(
                    name=merge_task.name,
                    muscle_type=muscle_type,
                    state_type=state_type,
                    muscle_group=muscle_group_name,
                    origin_position=ViaPointReal(
                        name=f"origin_{merge_task.name}",
                        parent_name=self.merged_model.muscle_groups[muscle_group_name].origin_parent_name,
                        position=origin_position_value,
                    ),
                    insertion_position=ViaPointReal(
                        name=f"insertion_{merge_task.name}",
                        parent_name=self.merged_model.muscle_groups[muscle_group_name].insertion_parent_name,
                        position=insertion_position_value,
                    ),
                    optimal_length=optimal_length_value,
                    maximal_force=maximal_force_value,
                    tendon_slack_length=tendon_slack_length_value,
                    pennation_angle=pennation_angle_value,
                    maximal_velocity=maximal_velocity_value,
                    maximal_excitation=maximal_excitation_value,
                )
            )
            for name, via in via_points.items():
                self.merged_model.muscle_groups[muscle_group_name].muscles[merge_task.name].add_via_point(
                    ViaPointReal(
                        name=name,
                        parent_name=via["parent"],
                        position=np.mean(np.array(via["position"]), axis=0),
                    )
                )

            # Remove the original muscles
            for muscle_name in merge_task.muscle_names:
                self.merged_model.muscle_groups[muscle_group_name].remove_muscle(muscle_name)

        return self.merged_model
