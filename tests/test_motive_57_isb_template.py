from pathlib import Path

import numpy as np

from biobuddy import Rotations, Translations
from biobuddy.components.generic.rigidbody.axis import Axis
from biobuddy.gui.c3d_creation_workflow import (
    c3d_workflow_draft,
    validate_c3d_workflow_draft,
)
from biobuddy.gui.c3d_model_creation import (
    C3dModelPreset,
    create_model_from_c3d_folder,
    default_static_virtual_points_for_c3d_model_preset,
)
from biobuddy.gui.model_builder import (
    FunctionalAxisProjectionPointSpec,
    FunctionalAxisSpec,
    FunctionalCenterSpec,
    MarkerEndpointSpec,
    build_generic_model,
)
from biobuddy.gui.motive_57_isb_template import motive_57_isb_template


def test_motive_57_isb_template_is_a_distinct_full_body_profile():
    template = motive_57_isb_template(use_functional=True)
    model = build_generic_model(template)

    assert template.name == "BioBuddy Motive (57) ISB from calibration C3D (SCoRE/SARA)"
    assert template.root_segment_name == "Pelvis"
    assert len(template.segments) == 15
    assert model.segments["Pelvis"].translations == Translations.XYZ
    assert model.segments["Pelvis"].rotations == Rotations.ZXY
    assert model.segments["RUpperArm"].rotations == Rotations.YXY
    assert model.segments["LUpperArm"].rotations == Rotations.YXY
    for side in ("L", "R"):
        for joint_name in ("HipJoint", "KneeJoint", "AnkleJoint"):
            assert model.segments[f"{side}{joint_name}"].rotations == Rotations.ZXY
        assert model.segments[f"{side}Thigh"].rotations == Rotations.NONE
        assert model.segments[f"{side}Shank"].rotations == Rotations.NONE
        assert model.segments[f"{side}Foot"].rotations == Rotations.NONE
        assert model.segments[f"{side}Thigh"].parent_name == f"{side}HipJoint"
        assert model.segments[f"{side}Shank"].parent_name == f"{side}KneeJoint"
        assert model.segments[f"{side}Foot"].parent_name == f"{side}AnkleJoint"
        assert model.segments[f"{side}Forearm"].rotations == Rotations.ZXY
        assert model.segments[f"{side}Hand"].rotations == Rotations.ZXY


def test_motive_57_isb_pelvis_and_thorax_use_isb_landmarks():
    segments = {segment.name: segment for segment in motive_57_isb_template().segments}

    pelvis = segments["Pelvis"].frame
    assert pelvis.origin.marker_names == ("LIAS", "RIAS")
    assert pelvis.first_axis.name == Axis.Name.Z
    assert pelvis.first_axis.start.marker_names == ("LIAS",)
    assert pelvis.first_axis.end.marker_names == ("RIAS",)
    assert pelvis.second_axis.name == Axis.Name.X
    assert pelvis.second_axis.start.marker_names == ("LIPS", "RIPS")
    assert pelvis.second_axis.end.marker_names == ("LIAS", "RIAS")
    assert pelvis.axis_to_keep == Axis.Name.Z

    thorax = segments["Thorax"].frame
    assert thorax.origin.marker_names == ("SJN",)
    assert thorax.first_axis.name == Axis.Name.Y
    assert thorax.first_axis.start.marker_names == ("SXS", "TV7")
    assert thorax.first_axis.end.marker_names == ("SJN", "CV7")
    assert "TV2" not in thorax.first_axis.end.marker_names
    assert thorax.axis_to_keep == Axis.Name.Y


def test_motive_57_isb_uses_static_anatomical_axes_and_functional_joint_centers():
    segments = {segment.name: segment for segment in motive_57_isb_template(use_functional=True).segments}

    for side in ("L", "R"):
        thigh_spec = segments[f"{side}Thigh"]
        shank_spec = segments[f"{side}Shank"]
        foot_spec = segments[f"{side}Foot"]
        thigh = thigh_spec.frame
        shank = shank_spec.frame
        foot = foot_spec.frame

        assert isinstance(shank_spec.joint_frame.origin, FunctionalAxisProjectionPointSpec)
        assert shank_spec.joint_frame.origin.method.value == "sara_direction"
        assert isinstance(shank_spec.joint_frame.second_axis, FunctionalAxisSpec)
        assert shank_spec.joint_frame.second_axis.method.value == "sara_direction"
        assert shank_spec.joint_frame.axis_to_keep == Axis.Name.Z
        assert isinstance(shank.origin, MarkerEndpointSpec)
        assert shank.origin.marker_names == (f"{side}FAL", f"{side}TAM")
        assert not isinstance(thigh.second_axis, FunctionalAxisSpec)
        assert not isinstance(shank.second_axis, FunctionalAxisSpec)
        assert shank.second_axis.name == Axis.Name.Z
        assert isinstance(foot_spec.joint_frame.origin, FunctionalCenterSpec)
        assert foot.origin.marker_names == (f"{side}FAL", f"{side}TAM")
        assert foot.first_axis.name == Axis.Name.Y
        assert foot.second_axis.name == Axis.Name.X
        assert foot.axis_to_keep == Axis.Name.Y

    assert segments["RThigh"].frame.second_axis.start.marker_names == ("RFME",)
    assert segments["RThigh"].frame.second_axis.end.marker_names == ("RFLE",)
    assert segments["LThigh"].frame.second_axis.start.marker_names == ("LFLE",)
    assert segments["LThigh"].frame.second_axis.end.marker_names == ("LFME",)
    assert segments["RShank"].frame.second_axis.start.marker_names == ("RTAM",)
    assert segments["RShank"].frame.second_axis.end.marker_names == ("RFAL",)
    assert segments["LShank"].frame.second_axis.start.marker_names == ("LFAL",)
    assert segments["LShank"].frame.second_axis.end.marker_names == ("LTAM",)


def test_motive_57_isb_workflow_exposes_joint_frames_without_marker_warnings():
    draft = c3d_workflow_draft(C3dModelPreset.MOTIVE_57_ISB)
    groups = {group.segment_name: group for group in draft.segment_marker_groups}
    settings = {setting.segment_name: setting for setting in draft.segment_settings}
    axes = {axis.name: axis for axis in draft.axes}

    for side in ("L", "R"):
        assert groups[f"{side}HipJoint"].parent_name == "Pelvis"
        assert groups[f"{side}Thigh"].parent_name == f"{side}HipJoint"
        assert groups[f"{side}KneeJoint"].parent_name == f"{side}Thigh"
        assert groups[f"{side}Shank"].parent_name == f"{side}KneeJoint"
        assert groups[f"{side}AnkleJoint"].parent_name == f"{side}Shank"
        assert groups[f"{side}Foot"].parent_name == f"{side}AnkleJoint"
        assert settings[f"{side}KneeJoint"].rotations == "zxy"
        assert settings[f"{side}Shank"].rotations == ""
        assert axes[f"{side}KneeJoint_second_axis"].method == "sara_direction"
        assert axes[f"{side}KneeJoint_second_axis"].axis == "z"
        assert axes[f"{side}KneeJoint_second_axis"].keep_vector

    issues = validate_c3d_workflow_draft(draft)
    assert not any("Joint' has no marker assigned" in issue.message for issue in issues)


def test_motive_57_isb_p6_lower_limb_frames_match_static_landmarks_and_sara():
    data_folder = Path(__file__).parents[1] / "examples" / "data" / "motive_57_p6"
    preset = C3dModelPreset.MOTIVE_57_ISB
    result = create_model_from_c3d_folder(
        data_folder,
        preset,
        static_virtual_points=default_static_virtual_points_for_c3d_model_preset(preset),
    )
    template = motive_57_isb_template(use_functional=True)
    laboratory_model = build_generic_model(template, functional_data=result.functional_data).to_real(result.static_data)
    pelvis_rotation = laboratory_model.segment_coordinate_system_in_global("Pelvis").rt_matrix[:3, :3]
    global_right = pelvis_rotation[:, 2]
    knee_directions = []

    for side in ("L", "R"):
        intermalleolar_center = np.nanmean(
            result.static_data.markers_center_position([f"{side}FAL", f"{side}TAM"])[:3, :],
            axis=1,
        )
        shank_frame = laboratory_model.segment_coordinate_system_in_global(f"{side}Shank").rt_matrix
        foot_frame = laboratory_model.segment_coordinate_system_in_global(f"{side}Foot").rt_matrix
        hip_joint_frame = laboratory_model.segment_coordinate_system_in_global(f"{side}HipJoint").rt_matrix
        ankle_joint_frame = laboratory_model.segment_coordinate_system_in_global(f"{side}AnkleJoint").rt_matrix
        knee_joint_frame = laboratory_model.segment_coordinate_system_in_global(f"{side}KneeJoint").rt_matrix

        np.testing.assert_allclose(shank_frame[:3, 3], intermalleolar_center, atol=1e-10)
        np.testing.assert_allclose(foot_frame[:3, 3], intermalleolar_center, atol=1e-10)
        np.testing.assert_allclose(hip_joint_frame[:3, :3], pelvis_rotation, atol=1e-10)
        np.testing.assert_allclose(ankle_joint_frame[:3, :3], shank_frame[:3, :3], atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(knee_joint_frame[:3, :3]), 1.0, atol=1e-10)
        assert float(np.dot(shank_frame[:3, 2], global_right)) > 0.8
        assert float(np.dot(foot_frame[:3, 2], global_right)) > 0.8

        expected_start = f"{side}FME" if side == "R" else f"{side}FLE"
        expected_end = f"{side}FLE" if side == "R" else f"{side}FME"
        expected_axis = np.nanmean(
            result.static_data.markers_center_position([expected_end])[:3, :]
            - result.static_data.markers_center_position([expected_start])[:3, :],
            axis=1,
        )
        expected_axis /= np.linalg.norm(expected_axis)
        knee_direction = knee_joint_frame[:3, 2]
        deviation = np.degrees(np.arccos(np.clip(abs(float(np.dot(knee_direction, expected_axis))), -1.0, 1.0)))
        assert deviation <= 30.0
        knee_directions.append(knee_direction)

    assert float(np.dot(knee_directions[0], knee_directions[1])) > 0.9
