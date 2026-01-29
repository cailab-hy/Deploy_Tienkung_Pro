#!/usr/bin/env python3
"""Extract ONNX graph outputs (default: actions only).

Example:
  ./extract_onnx_actions.py \
    --input ../config/policy_omni/omni_movebox.onnx \
    --output ../config/policy_omni/omni_movebox_actions.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import utils


def _default_output_path(src: Path, outputs: list[str]) -> Path:
    if outputs == ["actions"]:
        return src.with_name(f"{src.stem}_actions.onnx")
    return src.with_name(f"{src.stem}_extracted.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ONNX outputs (default: actions only).")
    parser.add_argument("-i", "--input", required=True, help="Input ONNX path")
    parser.add_argument("-o", "--output", default=None, help="Output ONNX path")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["obs", "time_step"],
        help="Input tensor names to keep (default: obs time_step)",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["actions"],
        help="Output tensor names to keep (default: actions)",
    )
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        parser.error(f"input ONNX not found: {src}")

    dst = Path(args.output) if args.output else _default_output_path(src, args.outputs)

    # Validate that requested names exist to give clearer errors.
    model = onnx.load(str(src))
    graph_inputs = {i.name for i in model.graph.input}
    graph_outputs = {o.name for o in model.graph.output}

    missing_inputs = [name for name in args.inputs if name not in graph_inputs]
    missing_outputs = [name for name in args.outputs if name not in graph_outputs]
    if missing_inputs:
        parser.error(f"unknown input name(s): {', '.join(missing_inputs)}")
    if missing_outputs:
        parser.error(f"unknown output name(s): {', '.join(missing_outputs)}")

    utils.extract_model(str(src), str(dst), input_names=args.inputs, output_names=args.outputs)
    print(f"[OK] Wrote: {dst}")
    print(f"[OK] Inputs: {', '.join(args.inputs)}")
    print(f"[OK] Outputs: {', '.join(args.outputs)}")


if __name__ == "__main__":
    main()
