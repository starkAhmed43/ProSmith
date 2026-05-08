from pathlib import Path
import shutil


DURABLE_TRAINING_FILES = [
    "bestmodel.pth",
    "final_results_val.csv",
    "final_results_test.csv",
    "results_val.csv",
    "results_test.csv",
    "run_summary.csv",
    "logfile.csv",
    "run_state.json",
    "time_running.dat",
]


def _expand(path_like):
    return Path(path_like).expanduser()


def _is_relative_to(path: Path, root: Path):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def map_to_scratch(path_like, scratch_root, home_root):
    path = _expand(path_like)
    scratch_root = _expand(scratch_root)
    home_root = _expand(home_root)

    if path.is_absolute():
        if _is_relative_to(path, home_root):
            return scratch_root / path.relative_to(home_root)
        return path
    return scratch_root / path


def map_to_home(path_like, scratch_root, home_root):
    path = _expand(path_like)
    scratch_root = _expand(scratch_root)
    home_root = _expand(home_root)

    if path.is_absolute():
        if _is_relative_to(path, scratch_root):
            return home_root / path.relative_to(scratch_root)
        return path
    return home_root / path


def sync_selected_files(src_dir, dst_dir, filenames):
    src_dir = _expand(src_dir)
    dst_dir = _expand(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, dst_dir / filename)


def sync_tree(src_dir, dst_dir):
    src_dir = _expand(src_dir)
    dst_dir = _expand(dst_dir)
    if not src_dir.exists():
        return
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
