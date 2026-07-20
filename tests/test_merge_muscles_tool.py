import numpy as np
import numpy.testing as npt
import pytest

from biobuddy import (
    BiomechanicalModelReal,
    MergeMusclesTool,
    MuscleMerge,
    MuscleGroupReal,
    MuscleReal,
    MuscleType,
    MuscleStateType,
    SegmentReal,
    ViaPointReal,
)
from biobuddy.components.via_point_utils import PathPointCondition, PathPointMovement
from biobuddy.components.functions import SimmSpline
from test_utils import create_model_with_two_muscles, create_muscle

# ---------- MuscleMerge init ----------


def test_muscle_merge_init_with_list():
    mm = MuscleMerge(name="merged", muscle_names=["m1", "m2"])
    assert mm.name == "merged"
    assert mm.muscle_names == ["m1", "m2"]


def test_muscle_merge_init_with_string():
    mm = MuscleMerge(name="merged", muscle_names="m1")
    assert mm.muscle_names == ["m1"]


def test_muscle_merge_init_invalid_type():
    with pytest.raises(ValueError, match="add_muscles expects a list"):
        MuscleMerge(name="merged", muscle_names=42)


# ---------- MergeMusclesTool.merge ----------


def test_merge_muscles_basic():

    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="seg_origin"))
    model.add_segment(SegmentReal(name="seg_insertion"))

    group = MuscleGroupReal(
        name="grp",
        origin_parent_name="seg_origin",
        insertion_parent_name="seg_insertion",
    )
    pos = np.array([[0.0], [0.0], [0.0]])

    # First muscle
    name = "m1"
    origin_pos = pos
    insertion_pos = pos + 1
    muscle_1 = MuscleReal(
        name=name,
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="grp",
        origin_position=ViaPointReal(name=f"origin_{name}", parent_name="seg_origin", position=origin_pos),
        insertion_position=ViaPointReal(name=f"insertion_{name}", parent_name="seg_insertion", position=insertion_pos),
        optimal_length=0.1,
        maximal_force=100.0,
        tendon_slack_length=0.05,
        pennation_angle=0.0,
        maximal_velocity=10.0,
        maximal_excitation=1.0,
    )
    muscle_1.add_via_point(ViaPointReal(name=f"V1_{name}", parent_name="seg_origin", position=pos - 0.5))
    muscle_1.add_via_point(ViaPointReal(name=f"V2_{name}", parent_name="seg_insertion", position=pos + 0.2))

    # Second muscle
    name = "m2"
    origin_pos = pos + 0.1
    insertion_pos = pos + 1.1
    muscle_2 = MuscleReal(
        name=name,
        muscle_type=MuscleType.HILL,
        state_type=MuscleStateType.DEGROOTE,
        muscle_group="grp",
        origin_position=ViaPointReal(name=f"origin_{name}", parent_name="seg_origin", position=origin_pos),
        insertion_position=ViaPointReal(name=f"insertion_{name}", parent_name="seg_insertion", position=insertion_pos),
        optimal_length=0.3,
        maximal_force=110.0,
        tendon_slack_length=0.055,
        pennation_angle=0.5,
        maximal_velocity=15.0,
        maximal_excitation=2.0,
    )
    muscle_2.add_via_point(ViaPointReal(name=f"V1_{name}", parent_name="seg_origin", position=pos - 0.25))
    muscle_2.add_via_point(ViaPointReal(name=f"V2_{name}", parent_name="seg_insertion", position=pos - 0.1))
    group.add_muscle(muscle_1)
    group.add_muscle(muscle_2)
    model.add_muscle_group(group)

    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    result = tool.merge()

    muscle_names = [m.name for mg in result.muscle_groups for m in mg.muscles]
    assert "merged" in muscle_names
    assert "m1" not in muscle_names
    assert "m2" not in muscle_names

    new_muscle_group = result.muscle_groups["grp"]

    assert len(new_muscle_group.muscles) == 1
    assert new_muscle_group.muscles[0].name == "merged"
    assert new_muscle_group.muscles[0].muscle_type == MuscleType.HILL
    assert new_muscle_group.muscles[0].state_type == MuscleStateType.DEGROOTE
    npt.assert_almost_equal(new_muscle_group.muscles[0].origin_position.position[:3], (pos + pos + 0.1) / 2)
    npt.assert_almost_equal(new_muscle_group.muscles[0].insertion_position.position[:3], (pos + 1 + pos + 1.1) / 2)
    assert new_muscle_group.muscles[0].optimal_length == (0.3 + 0.1) / 2
    assert new_muscle_group.muscles[0].maximal_force == (110 + 100)
    assert new_muscle_group.muscles[0].tendon_slack_length == (0.05 + 0.055) / 2
    assert new_muscle_group.muscles[0].pennation_angle == (0.0 + 0.5) / 2
    assert new_muscle_group.muscles[0].maximal_velocity == (15 + 10) / 2
    assert new_muscle_group.muscles[0].maximal_excitation == (1 + 2) / 2


def test_merge_muscles_force_is_sum():
    model = create_model_with_two_muscles(
        m1_kw={"maximal_force": 100.0},
        m2_kw={"maximal_force": 150.0},
    )
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    result = tool.merge()

    assert result.muscle_groups["grp"].muscles["merged"].maximal_force == pytest.approx(250.0)


def test_merge_muscles_position_is_mean():
    origin_a = np.array([[0.0], [0.0], [0.0]])
    origin_b = np.array([[1.0], [0.0], [0.0]])
    ins = np.array([[0.5], [0.5], [0.5]])
    model = create_model_with_two_muscles(m1_origin=origin_a, m2_origin=origin_b, m1_insert=ins, m2_insert=ins)
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    result = tool.merge()

    merged_origin = result.muscle_groups["grp"].muscles["merged"].origin_position.position
    npt.assert_allclose(merged_origin[:3], [[0.5], [0.0], [0.0]])


def test_merge_muscles_scalars_are_mean():
    model = create_model_with_two_muscles(
        m1_kw={
            "optimal_length": 0.1,
            "tendon_slack_length": 0.05,
            "pennation_angle": 0.0,
            "maximal_velocity": 10.0,
            "maximal_excitation": 1.0,
        },
        m2_kw={
            "optimal_length": 0.3,
            "tendon_slack_length": 0.055,
            "pennation_angle": 0.5,
            "maximal_velocity": 15.0,
            "maximal_excitation": 2.0,
        },
    )
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    result = tool.merge()

    m = result.muscle_groups["grp"].muscles["merged"]
    assert m.optimal_length == pytest.approx((0.1 + 0.3) / 2)
    assert m.tendon_slack_length == pytest.approx((0.05 + 0.055) / 2)
    assert m.pennation_angle == pytest.approx((0.0 + 0.5) / 2)
    assert m.maximal_velocity == pytest.approx((10.0 + 15.0) / 2)
    assert m.maximal_excitation == pytest.approx((1.0 + 2.0) / 2)


def test_merge_muscles_preserves_type_and_state():
    model = create_model_with_two_muscles()
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    result = tool.merge()

    m = result.muscle_groups["grp"].muscles["merged"]
    assert m.muscle_type == MuscleType.HILL
    assert m.state_type == MuscleStateType.DEGROOTE


def test_merge_muscles_unknown_muscle_raises():
    model = create_model_with_two_muscles()
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "ghost"]))

    with pytest.raises(ValueError, match="not found in the merged model"):
        tool.merge()


def test_merge_muscles_wrong_group_raises():
    model = create_model_with_two_muscles()
    # Move m2 to a different group so it no longer belongs to "grp"
    model.muscle_groups["grp"].muscles["m2"]._muscle_group = "other_grp"
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))

    with pytest.raises(ValueError, match="does not belong to muscle group"):
        tool.merge()


def test_merge_muscles_does_not_mutate_original():
    model = create_model_with_two_muscles()
    original_names = list(model.muscle_names)
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    tool.merge()

    assert list(model.muscle_names) == original_names


def test_merge_muscles_with_via_points():
    pos = np.array([[0.0], [0.0], [0.0]])
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="seg_origin"))
    model.add_segment(SegmentReal(name="seg_insertion"))
    group = MuscleGroupReal(name="grp", origin_parent_name="seg_origin", insertion_parent_name="seg_insertion")

    m1 = create_muscle("m1", pos, pos + 1, "grp")
    m1.add_via_point(ViaPointReal(name="V1_m1", parent_name="seg_origin", position=pos - 0.5))
    m2 = create_muscle("m2", pos + 0.1, pos + 1.1, "grp")
    m2.add_via_point(ViaPointReal(name="V1_m2", parent_name="seg_origin", position=pos - 0.25))
    group.add_muscle(m1)
    group.add_muscle(m2)
    model.add_muscle_group(group)

    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    result = tool.merge()

    assert result.muscle_groups["grp"].muscles["merged"].nb_via_points == 1
    npt.assert_allclose(
        result.muscle_groups["grp"].muscles["merged"].via_points[0].position[:3],
        (pos - 0.5 + pos - 0.25) / 2,
    )


# ---------- get_muscle_muscle_group_name ----------


def test_get_muscle_muscle_group_name_found():
    model = create_model_with_two_muscles()
    assert model.get_muscle_muscle_group_name("m1") == "grp"
    assert model.get_muscle_muscle_group_name("m2") == "grp"


def test_get_muscle_muscle_group_name_multiple_groups():
    model = BiomechanicalModelReal()
    model.add_segment(SegmentReal(name="seg_a"))
    model.add_segment(SegmentReal(name="seg_b"))
    model.add_segment(SegmentReal(name="seg_c"))

    pos = np.array([[0.0], [0.0], [0.0]])
    grp1 = MuscleGroupReal(name="grp1", origin_parent_name="seg_a", insertion_parent_name="seg_b")
    grp1.add_muscle(create_muscle("alpha", pos, pos + 1, "grp1"))
    model.add_muscle_group(grp1)

    grp2 = MuscleGroupReal(name="grp2", origin_parent_name="seg_b", insertion_parent_name="seg_c")
    grp2.add_muscle(create_muscle("beta", pos, pos + 1, "grp2"))
    model.add_muscle_group(grp2)

    assert model.get_muscle_muscle_group_name("alpha") == "grp1"
    assert model.get_muscle_muscle_group_name("beta") == "grp2"


def test_get_muscle_muscle_group_name_not_found():
    model = create_model_with_two_muscles()
    with pytest.raises(ValueError, match="not found in the model"):
        model.get_muscle_muscle_group_name("ghost")


# ---------- MergeMusclesTool raise conditions ----------


def _add_conditional_via_point(muscle: MuscleReal, name: str) -> None:
    vp = ViaPointReal(
        name=name,
        parent_name="seg_origin",
        position=np.array([[0.0], [0.0], [0.0]]),
        condition=PathPointCondition(dof_name="dof1", range_min=-1.0, range_max=1.0),
    )
    muscle.add_via_point(vp)


def _add_moving_via_point(muscle: MuscleReal, name: str) -> None:
    spline = SimmSpline(x_points=np.array([0.0, 1.0]), y_points=np.array([0.0, 1.0]))
    vp = ViaPointReal(
        name=name,
        parent_name="seg_origin",
        movement=PathPointMovement(
            dof_names=["dof1", "dof2", "dof3"],
            locations=[spline, spline, spline],
        ),
    )
    muscle.add_via_point(vp)


def test_merge_raises_conditional_via_point_on_first_muscle():
    model = create_model_with_two_muscles()
    _add_conditional_via_point(model.muscle_groups["grp"].muscles["m1"], "cond_vp")
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    with pytest.raises(NotImplementedError, match="Conditional via points not implemented"):
        tool.merge()


def test_merge_raises_moving_via_point_on_first_muscle():
    model = create_model_with_two_muscles()
    _add_moving_via_point(model.muscle_groups["grp"].muscles["m1"], "move_vp")
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    with pytest.raises(NotImplementedError, match="Moving via points not implemented"):
        tool.merge()


def test_merge_raises_mismatched_muscle_type():
    model = create_model_with_two_muscles()
    model.muscle_groups["grp"].muscles["m2"]._muscle_type = MuscleType.HILL_THELEN
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    with pytest.raises(ValueError, match="muscle type of the muscles is not all the same"):
        tool.merge()


def test_merge_raises_mismatched_state_type():
    model = create_model_with_two_muscles()
    model.muscle_groups["grp"].muscles["m2"]._state_type = MuscleStateType.BUCHANAN
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    with pytest.raises(ValueError, match="muscle state type of the muscles is not all the same"):
        tool.merge()


def test_merge_raises_mismatched_via_point_count():
    pos = np.array([[0.0], [0.0], [0.0]])
    model = create_model_with_two_muscles()
    # m1 gets one via point, m2 gets none
    model.muscle_groups["grp"].muscles["m1"].add_via_point(
        ViaPointReal(name="vp_m1", parent_name="seg_origin", position=pos)
    )
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    with pytest.raises(ValueError, match="not the same number of via points"):
        tool.merge()


def test_merge_raises_conditional_via_point_on_subsequent_muscle():
    pos = np.array([[0.0], [0.0], [0.0]])
    model = create_model_with_two_muscles()
    # Both muscles need a via point so the count check passes, but m2's is conditional
    model.muscle_groups["grp"].muscles["m1"].add_via_point(
        ViaPointReal(name="vp_m1", parent_name="seg_origin", position=pos)
    )
    _add_conditional_via_point(model.muscle_groups["grp"].muscles["m2"], "cond_vp_m2")
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    with pytest.raises(NotImplementedError, match="Conditional via points not implemented"):
        tool.merge()


def test_merge_raises_moving_via_point_on_subsequent_muscle():
    pos = np.array([[0.0], [0.0], [0.0]])
    model = create_model_with_two_muscles()
    model.muscle_groups["grp"].muscles["m1"].add_via_point(
        ViaPointReal(name="vp_m1", parent_name="seg_origin", position=pos)
    )
    _add_moving_via_point(model.muscle_groups["grp"].muscles["m2"], "move_vp_m2")
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    with pytest.raises(NotImplementedError, match="Moving via points not implemented"):
        tool.merge()


def test_merge_raises_unmatched_via_point_parent():
    pos = np.array([[0.0], [0.0], [0.0]])
    model = create_model_with_two_muscles()
    # m1 has a via point on seg_origin; m2 has one on seg_insertion — no common parent to match
    model.muscle_groups["grp"].muscles["m1"].add_via_point(
        ViaPointReal(name="vp_m1", parent_name="seg_origin", position=pos)
    )
    model.muscle_groups["grp"].muscles["m2"].add_via_point(
        ViaPointReal(name="vp_m2", parent_name="seg_insertion", position=pos)
    )
    tool = MergeMusclesTool(model)
    tool.add(MuscleMerge(name="merged", muscle_names=["m1", "m2"]))
    with pytest.raises(ValueError, match="could not be matched with any via points"):
        tool.merge()
