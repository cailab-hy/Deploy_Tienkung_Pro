import argparse
from pathlib import Path

import numpy as np
import yaml


def _to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _load_npz(npz_path: Path):
    try:
        return np.load(npz_path, allow_pickle=False)
    except ValueError:
        # Some npz files store strings as object arrays; retry with pickle enabled.
        return np.load(npz_path, allow_pickle=True)


def _resolve_ref_body_index(data, ref_body_name: str | None, ref_body_index: int | None) -> int:
    if ref_body_name and "body_names" in data.files:
        body_names = data["body_names"]
        try:
            names = [str(x) for x in body_names.tolist()]
        except Exception:
            names = [str(x) for x in body_names]
        if ref_body_name in names:
            return names.index(ref_body_name)

    if ref_body_index is not None:
        return ref_body_index

    raise ValueError(
        "Unable to resolve ref body index. Provide --ref-body-name that exists in body_names or --ref-body-index."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert motion npz to YAML (joint_pos/vel + ref body quat).")
    parser.add_argument("--npz", required=True, help="Input npz file")
    parser.add_argument("--yaml", required=True, help="Output yaml file")
    parser.add_argument(
        "--ref-body-name",
        default="body_yaw_link",
        help="Reference body name to extract from body_quat_w (default: body_yaw_link)",
    )
    parser.add_argument(
        "--ref-body-index",
        type=int,
        default=None,
        help="Fallback reference body index if body_names is missing",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    yaml_path = Path(args.yaml)

    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    data = _load_npz(npz_path)

    out = {}
    ref_body_index = None
    if "body_quat_w" in data.files:
        ref_body_index = _resolve_ref_body_index(data, args.ref_body_name, args.ref_body_index)

    for k in data.files:
        arr = data[k]
        if k == "joint_pos":
            if arr.ndim != 2 or arr.shape[1] < 7:
                raise ValueError(f"joint_pos shape unexpected: {arr.shape}")
            out[k] = _to_python(arr[:, 7:])
        elif k == "joint_vel":
            if arr.ndim != 2 or arr.shape[1] < 6:
                raise ValueError(f"joint_vel shape unexpected: {arr.shape}")
            out[k] = _to_python(arr[:, 6:])
        elif k == "body_quat_w":
            if arr.ndim != 3 or arr.shape[2] != 4:
                raise ValueError(f"body_quat_w shape unexpected: {arr.shape}")
            if ref_body_index is None or ref_body_index >= arr.shape[1]:
                raise ValueError(
                    f"ref body index out of range: {ref_body_index} (num bodies: {arr.shape[1]})"
                )
            out[k] = _to_python(arr[:, ref_body_index])
        print(k, arr.shape)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

    print(f"Saved: {yaml_path}")
    print("Keys:", data.files)


if __name__ == "__main__":
    main()
