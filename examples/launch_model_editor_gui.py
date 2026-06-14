"""
This example shows how to launch the model editor GUI.
"""

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

from biobuddy import launch_model_editor

LOWER_LIMB_CALIBRATION_FOLDER = Path(__file__).resolve().parent / "data" / "lower_limb_calibration"
FULL_BODY_MODEL202_FOLDER = Path(__file__).resolve().parent / "data" / "full_body_model202"


def _existing_folder(path: str) -> Path:
    """
    Return an existing folder path or stop argument parsing with a clear message.
    """
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise ArgumentTypeError(f"C3D folder does not exist: {folder}")
    return folder


def _parse_arguments():
    parser = ArgumentParser(description="Launch the BioBuddy model editor GUI.")
    parser.add_argument(
        "--new-from-c3d",
        action="store_true",
        help="Open the New from C3D workflow as soon as the GUI starts.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Preset to preload in the New from C3D workflow, for example lower-limbs-functional.",
    )
    parser.add_argument(
        "--c3d-folder",
        default=None,
        help="Folder containing the C3D calibration files to preload.",
    )
    parser.add_argument(
        "--lower-limbs-functional-example",
        action="store_true",
        help=(
            "Shortcut for --new-from-c3d --preset lower-limbs-functional "
            "--c3d-folder examples/data/lower_limb_calibration."
        ),
    )
    parser.add_argument(
        "--full-body-model202-example",
        action="store_true",
        help="Shortcut for --new-from-c3d --preset full-body --c3d-folder examples/data/full_body_model202.",
    )
    return parser.parse_args()


def launch_gui():
    args = _parse_arguments()
    preset = args.preset
    c3d_folder = args.c3d_folder
    open_new_from_c3d = args.new_from_c3d

    if args.lower_limbs_functional_example:
        open_new_from_c3d = True
        preset = "lower-limbs-functional"
        c3d_folder = str(LOWER_LIMB_CALIBRATION_FOLDER)
    if args.full_body_model202_example:
        open_new_from_c3d = True
        preset = "full-body"
        c3d_folder = str(FULL_BODY_MODEL202_FOLDER)

    if c3d_folder is not None:
        c3d_folder = _existing_folder(c3d_folder)

    launch_model_editor(
        open_new_from_c3d=open_new_from_c3d,
        c3d_preset=preset,
        c3d_folder=c3d_folder,
    )


if __name__ == "__main__":
    launch_gui()
