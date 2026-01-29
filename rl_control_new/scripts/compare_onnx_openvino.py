#!/usr/bin/env python3
"""Compare ONNX Runtime vs OpenVINO outputs on the same inputs.

Example:
  python3 compare_onnx_openvino.py     --onnx ../config/policy_omni/omni_movebox_actions.onnx     --ov-xml ../config/policy_omni/omni_movebox_actions.xml     --obs-dim 159     --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"onnxruntime import failed: {exc}")

try:
    from openvino.runtime import Core, Tensor
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"openvino import failed: {exc}")


ISAAC_TO_MUJOCO = [
    1, 6, 11, 16, 20, 24, 2, 7, 12, 17,
    21, 25, 0, 3, 8, 13, 4, 9, 14, 18,
    22, 26, 28, 5, 10, 15, 19, 23, 27, 29,
]

DEFAULT_DOF_POS = np.array(
    [
        0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
        0.0, -0.5, 0.0, 1.0, -0.5, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.1, 0.0, -0.3, 0.0, 0.0, 0.0,
        0.0, -0.1, -0.0, -0.3, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)



def _build_obs_from_refer(refer_yaml: Path, frame: int) -> np.ndarray:
    with open(refer_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    joint_pos = np.asarray(data["joint_pos"][frame], dtype=np.float32)
    joint_vel = np.asarray(data["joint_vel"][frame], dtype=np.float32)

    if joint_pos.shape[0] != DEFAULT_DOF_POS.shape[0]:
        raise ValueError(f"joint_pos dim {joint_pos.shape[0]} != {DEFAULT_DOF_POS.shape[0]}")
    if joint_vel.shape[0] != DEFAULT_DOF_POS.shape[0]:
        raise ValueError(f"joint_vel dim {joint_vel.shape[0]} != {DEFAULT_DOF_POS.shape[0]}")

    motion_command = np.concatenate([joint_pos, joint_vel], axis=0)
    motion_ref_ori_b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    base_ang_vel = np.zeros(3, dtype=np.float32)
    dof_pos = joint_pos - DEFAULT_DOF_POS
    dof_vel = joint_vel
    prev_actions = np.zeros_like(joint_pos, dtype=np.float32)

    obs = np.concatenate(
        [motion_command, motion_ref_ori_b, base_ang_vel, dof_pos, dof_vel, prev_actions], axis=0
    )
    return obs.reshape(1, -1)

def _load_obs(args) -> np.ndarray:
    if args.obs_npy:
        obs = np.load(args.obs_npy)
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        return obs

    rng = np.random.default_rng(args.seed)
    obs = rng.uniform(low=-1.0, high=1.0, size=(1, args.obs_dim)).astype(np.float32)
    return obs


def _pick_output_name(names: List[str]) -> str:
    for candidate in ("actions", "output", "output0"):
        if candidate in names:
            return candidate
    return names[0]


def _make_time_input(shape, value: float) -> np.ndarray:
    if not shape:
        return np.array([value], dtype=np.float32)
    return np.full(shape, value, dtype=np.float32)


def _run_onnx(path: Path, obs: np.ndarray, time_step: float) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    input_shapes = [i.shape for i in sess.get_inputs()]

    feeds = {}
    for name, shape in zip(input_names, input_shapes):
        if "obs" in name:
            feeds[name] = obs
        elif "time" in name or "step" in name:
            feeds[name] = _make_time_input(shape, time_step)
        else:
            # fallback: first input = obs, second = time_step
            if not feeds:
                feeds[name] = obs
            else:
                feeds[name] = _make_time_input(shape, time_step)

    outputs = [o.name for o in sess.get_outputs()]
    out_name = _pick_output_name(outputs)
    out = sess.run([out_name], feeds)[0]
    return np.asarray(out, dtype=np.float32)


def _run_openvino(xml_path: Path, obs: np.ndarray, time_step: float) -> np.ndarray:
    core = Core()
    model = core.read_model(model=str(xml_path))
    compiled = core.compile_model(model, "CPU")
    infer = compiled.create_infer_request()

    inputs = compiled.inputs
    if not inputs:
        raise RuntimeError("OpenVINO model has no inputs")

    for i, ov_input in enumerate(inputs):
        name = ov_input.any_name
        shape = ov_input.get_shape()
        if "obs" in name:
            infer.set_tensor(name, Tensor(obs))
        elif "time" in name or "step" in name:
            infer.set_tensor(name, Tensor(_make_time_input(shape, time_step)))
        else:
            if i == 0:
                infer.set_tensor(name, Tensor(obs))
            else:
                infer.set_tensor(name, Tensor(_make_time_input(shape, time_step)))

    infer.infer()
    outputs = compiled.outputs
    out_name = _pick_output_name([o.any_name for o in outputs])
    out_tensor = infer.get_tensor(out_name)
    return np.asarray(out_tensor.data, dtype=np.float32)


def _parse_urdf_limits(urdf_path: Path) -> dict:
    from xml.etree import ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    limits = {}
    ordered_names = []
    for joint in root.iter("joint"):
        jtype = joint.get("type")
        name = joint.get("name")
        if jtype == "fixed" or name is None:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = float(limit.get("lower"))
        upper = float(limit.get("upper"))
        limits[name] = (lower, upper)
        ordered_names.append(name)
    return {"limits": limits, "order": ordered_names}


def _load_joint_names_from_npz(npz_path: Path) -> List[str]:
    data = np.load(npz_path, allow_pickle=True)
    if "joint_names" not in data.files:
        raise ValueError("joint_names not found in npz")
    names = data["joint_names"].tolist()
    return [str(n) for n in names]


def _check_limits(
    actions: np.ndarray,
    urdf_path: Path,
    joint_names_npz: Path | None,
    action_scale: float,
    reorder: bool,
) -> None:
    if actions.ndim == 2:
        actions_vec = actions[0]
    else:
        actions_vec = actions

    if actions_vec.shape[0] != 30:
        print(f"[Limits] Skip: actions dim {actions_vec.shape[0]} != 30")
        return

    if reorder:
        actions_vec = actions_vec[ISAAC_TO_MUJOCO]

    q_target = actions_vec * action_scale + DEFAULT_DOF_POS

    parsed = _parse_urdf_limits(urdf_path)
    limits = parsed["limits"]
    joint_order = parsed["order"]

    if joint_names_npz:
        joint_order = _load_joint_names_from_npz(joint_names_npz)

    violations = []
    for idx, name in enumerate(joint_order[: q_target.shape[0]]):
        if name not in limits:
            continue
        lower, upper = limits[name]
        val = float(q_target[idx])
        if val < lower or val > upper:
            violations.append((idx, name, val, lower, upper))

    print(f"[Limits] checked {len(joint_order[: q_target.shape[0]])} joints")
    print(f"[Limits] violations: {len(violations)}")
    for idx, name, val, lower, upper in violations[:10]:
        print(f"  - {idx:02d} {name}: {val:.4f} (limit {lower:.4f}..{upper:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ONNX Runtime and OpenVINO outputs.")
    parser.add_argument("--onnx", required=True, help="ONNX path")
    parser.add_argument("--ov-xml", required=True, help="OpenVINO XML path")
    parser.add_argument("--obs-npy", default=None, help="Optional obs .npy (shape [1,dim])")
    parser.add_argument("--refer-yaml", default=None, help="refer.yaml to build obs from")
    parser.add_argument("--frame", type=int, default=0, help="Frame index when using refer.yaml")
    parser.add_argument("--obs-dim", type=int, default=159, help="Obs dim when using random input")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for obs")
    parser.add_argument("--time-step", type=float, default=0.0, help="time_step input value")
    parser.add_argument("--check-limits", action="store_true", help="Check joint limits")
    parser.add_argument("--urdf", default=None, help="URDF path for limits")
    parser.add_argument("--joint-names-npz", default=None, help="npz with joint_names for limit ordering")
    parser.add_argument("--action-scale", type=float, default=0.25, help="Action scale")
    parser.add_argument("--no-reorder", action="store_true", help="Disable ISAAC->MuJoCo reorder")
    args = parser.parse_args()

    obs = _load_obs(args)
    onnx_out = _run_onnx(Path(args.onnx), obs, args.time_step)
    ov_out = _run_openvino(Path(args.ov_xml), obs, args.time_step)

    diff = np.abs(onnx_out - ov_out)
    print("[Compare] ONNX output shape:", onnx_out.shape)
    print("[Compare] OpenVINO output shape:", ov_out.shape)
    print("[Compare] max abs diff:", float(diff.max()))
    print("[Compare] mean abs diff:", float(diff.mean()))
    print("[Compare] rms diff:", float(np.sqrt((diff ** 2).mean())))

    if args.check_limits:
        if not args.urdf:
            raise SystemExit("--urdf is required when --check-limits is set")
        _check_limits(
            ov_out,
            Path(args.urdf),
            Path(args.joint_names_npz) if args.joint_names_npz else None,
            args.action_scale,
            reorder=not args.no_reorder,
        )


if __name__ == "__main__":
    main()
