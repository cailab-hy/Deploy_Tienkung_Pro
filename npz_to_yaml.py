import numpy as np
import yaml
import argparse

def to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main(npz_path, yaml_path, body_index=22):
    data = np.load(npz_path, allow_pickle=False)

    out = {}
    for k in data.files:  # npz 안의 key 목록
        arr = data[k]
        if k in ["joint_pos"]:
            out[k] = to_python(arr[:,7:])
        elif k in ["joint_vel"]:
            out[k] = to_python(arr[:,6:])
        elif k in ["body_pos_w"]:
            out["ref_pos_xyz"] = to_python(arr[:, body_index])
        elif k in ["body_quat_w"]:
            quat_wxyz = arr[:, body_index]
            quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
            out["ref_quat_xyzw"] = to_python(quat_xyzw)
        print(k,arr.shape)
        
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

    print(f"Saved: {yaml_path}")
    print("Keys:", data.files)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path")
    parser.add_argument("yaml_path")
    parser.add_argument("--body-index", type=int, default=22)
    args = parser.parse_args()
    main(args.npz_path, args.yaml_path, body_index=args.body_index)
