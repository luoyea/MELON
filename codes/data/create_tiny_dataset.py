import argparse
import json
import os
import shutil
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def pick_users(train_data, n_users):
    user_ids = sorted(int(u) for u in train_data.keys())
    if n_users <= 0 or n_users >= len(user_ids):
        return set(user_ids)
    return set(user_ids[:n_users])


def filter_interactions(data, users, min_interactions=1):
    out = {}
    for uid in users:
        key = str(uid)
        if key not in data:
            continue
        items = data[key]
        if len(items) >= min_interactions:
            out[key] = items
    return out


def remap_items(split_dict, item_map):
    out = {}
    for u, items in split_dict.items():
        out[u] = [item_map[i] for i in items if i in item_map]
    return out


def main():
    parser = argparse.ArgumentParser(description="Create a tiny dataset split from an existing MELON dataset.")
    parser.add_argument("--src", type=str, required=True, help="Source dataset folder under codes/data, e.g. MenClothing")
    parser.add_argument("--dst", type=str, required=True, help="Target tiny dataset folder name, e.g. MenClothing_tiny")
    parser.add_argument("--n_users", type=int, default=500, help="Number of users to keep (by sorted user id)")
    parser.add_argument("--min_interactions", type=int, default=1, help="Minimum interactions kept per split")
    parser.add_argument("--copy_knn", type=int, default=0, help="Copy image_knn_*.txt/text_knn_*.txt if present (not recommended after remap)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, args.src)
    dst_dir = os.path.join(script_dir, args.dst)
    src_core = os.path.join(src_dir, "5-core")
    dst_core = os.path.join(dst_dir, "5-core")

    if not os.path.exists(src_core):
        raise FileNotFoundError("Source 5-core folder not found: {}".format(src_core))

    os.makedirs(dst_core, exist_ok=True)

    train_path = os.path.join(src_core, "train.json")
    val_path = os.path.join(src_core, "val.json")
    test_path = os.path.join(src_core, "test.json")

    train = load_json(train_path)
    val = load_json(val_path)
    test = load_json(test_path)

    users = pick_users(train, args.n_users)
    train_tiny_raw = filter_interactions(train, users, args.min_interactions)
    val_tiny_raw = filter_interactions(val, users, args.min_interactions)
    test_tiny_raw = filter_interactions(test, users, args.min_interactions)

    # Build compact item id map to avoid out-of-range indices in sparse matrices.
    item_set = set()
    for d in (train_tiny_raw, val_tiny_raw, test_tiny_raw):
        for _, items in d.items():
            item_set.update(items)
    old_items = sorted(item_set)
    item_map = {old_i: new_i for new_i, old_i in enumerate(old_items)}

    train_tiny = remap_items(train_tiny_raw, item_map)
    val_tiny = remap_items(val_tiny_raw, item_map)
    test_tiny = remap_items(test_tiny_raw, item_map)

    save_json(os.path.join(dst_core, "train.json"), train_tiny)
    save_json(os.path.join(dst_core, "val.json"), val_tiny)
    save_json(os.path.join(dst_core, "test.json"), test_tiny)

    copied = []
    # Subset modality features by remapped item ids.
    image_fp = os.path.join(src_dir, "image_feat.npy")
    text_fp = os.path.join(src_dir, "text_feat.npy")
    if os.path.exists(image_fp):
        image_feat = np.load(image_fp)
        image_feat_tiny = image_feat[old_items]
        np.save(os.path.join(dst_dir, "image_feat.npy"), image_feat_tiny.astype(np.float32))
        copied.append("image_feat.npy(remapped)")
    if os.path.exists(text_fp):
        text_feat = np.load(text_fp)
        text_feat_tiny = text_feat[old_items]
        np.save(os.path.join(dst_dir, "text_feat.npy"), text_feat_tiny.astype(np.float32))
        copied.append("text_feat.npy(remapped)")

    # Save map for traceability.
    with open(os.path.join(dst_dir, "item_id_map.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in item_map.items()}, f)

    if args.copy_knn == 1:
        for fn in os.listdir(src_dir):
            if fn.startswith("image_knn_") or fn.startswith("text_knn_"):
                src_fp = os.path.join(src_dir, fn)
                dst_fp = os.path.join(dst_dir, fn)
                if os.path.isfile(src_fp):
                    shutil.copy2(src_fp, dst_fp)
                    copied.append(fn)

    print("Create tiny dataset done.")
    print("src:", src_dir)
    print("dst:", dst_dir)
    print("users selected from train:", len(users))
    print("items remapped:", len(old_items))
    print("train users:", len(train_tiny), "train interactions:", sum(len(v) for v in train_tiny.values()))
    print("val users:", len(val_tiny), "val interactions:", sum(len(v) for v in val_tiny.values()))
    print("test users:", len(test_tiny), "test interactions:", sum(len(v) for v in test_tiny.values()))
    print("copied files:", copied)


if __name__ == "__main__":
    main()
