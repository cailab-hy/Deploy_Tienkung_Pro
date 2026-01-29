import numpy as np
import yaml
import argparse

def to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def main(npz_path, yaml_path):
    data = np.load(npz_path, allow_pickle=False)

    out = {}
    for k in data.files:  # npz 안의 key 목록
        arr = data[k]
        if k in ["joint_pos"]:
            out[k] = to_python(arr[:,7:])
        elif k in ["joint_vel"]:
            out[k] = to_python(arr[:,6:])
        elif k in ["body_quat_w"]:
            out[k] = to_python(arr[:,22])
        print(k,arr.shape)
        
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

    print(f"Saved: {yaml_path}")
    print("Keys:", data.files)
    
main("/home/cai/Downloads/sub3_largebox_003_mj_walker_obj.npz", "/home/cai/Downloads/refer.yaml")