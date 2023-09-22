import json
from pathlib import Path
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--size", type=int, default=40)
parser.add_argument("--suffix", type=str, default='mini-custom')
args = parser.parse_args()


def subsample_anet():
    dataset = "ActivityNet1.2"
    root_path = Path(f"./{dataset}-Annotations")
    data = np.load(f"./{dataset}-I3D-JOINTFeatures.npy",
                   encoding='bytes',
                   allow_pickle=True)
    video_name = np.load(str(root_path / "videoname.npy"), allow_pickle=True)
    classlist = np.load(str(root_path / "classlist.npy"), allow_pickle=True)
    segments = np.load(str(root_path / "segments.npy"), allow_pickle=True)
    labels = np.load(str(root_path / "labels.npy"), allow_pickle=True)
    labels_all = np.load(str(root_path / "labels_all.npy"), allow_pickle=True)
    subsets = np.load(str(root_path / "subset.npy"), allow_pickle=True)
    duration = np.load(str(root_path / "duration.npy"), allow_pickle=True)

    # randomly sample n classes
    np.random.seed(args.seed)

    selected_classes = np.random.choice(classlist,
                                        size=args.size,
                                        replace=False)
    inds = []
    _seg = []
    _labels = []
    _labels_all = []

    for i, (lab, seg) in enumerate(zip(labels, segments)):
        tmp_l = []
        tmp_s = []
        for k, each_l in enumerate(lab):
            if each_l.encode('utf8') in selected_classes:
                tmp_l.append(each_l)
                tmp_s.append(seg[k])

        if len(tmp_l) > 0:
            inds.append(i)
            _seg.append(tmp_s)
            _labels.append(tmp_l)
            _labels_all.append(np.unique(tmp_l))

    out_dataset = f"ActivityNet1.2-{args.suffix}"
    root_path = Path(f"./{out_dataset}-Annotations")
    root_path.mkdir(exist_ok=True)
    np.save(str(root_path / "videoname.npy"),
            video_name[inds],
            allow_pickle=True)
    np.save(str(root_path / "classlist.npy"),
            selected_classes,
            allow_pickle=True)
    np.save(str(root_path / "segments.npy"), np.array(_seg), allow_pickle=True)
    np.save(str(root_path / "labels.npy"),
            np.array(_labels),
            allow_pickle=True)
    np.save(str(root_path / "labels_all.npy"),
            np.array(_labels_all),
            allow_pickle=True)
    np.save(str(root_path / "subset.npy"), subsets[inds], allow_pickle=True)
    np.save(str(root_path / "duration.npy"), duration[inds], allow_pickle=True)
    np.save(f"./{out_dataset}-I3D-JOINTFeatures.npy",
            data[inds],
            allow_pickle=True)


def create_gt():
    dataset = f"ActivityNet1.2-{args.suffix}"
    root_path = Path(f"./{dataset}-Annotations")
    video_name = np.load(str(root_path / "videoname.npy"), allow_pickle=True)
    classlist = np.load(str(root_path / "classlist.npy"), allow_pickle=True)
    segments = np.load(str(root_path / "segments.npy"), allow_pickle=True)
    labels = np.load(str(root_path / "labels.npy"), allow_pickle=True)
    labels_all = np.load(str(root_path / "labels_all.npy"), allow_pickle=True)
    subsets = np.load(str(root_path / "subset.npy"), allow_pickle=True)
    duration = np.load(str(root_path / "duration.npy"), allow_pickle=True)

    data = {}
    for i, vn in enumerate(video_name):
        ann = []
        for _lab, _segm in zip(labels[i], segments[i]):
            ann.append({
                "segment": [f"{elem:.2f}" for elem in _segm],
                "label": str(_lab)
            })
        data[vn.decode()] = {
            "subset": "train" if subsets[i] == b'training' else "test",
            "annotations": ann
        }
    os.makedirs(dataset, exist_ok=True)

    dict_class = {
        counter: cls
        for counter, cls in enumerate(sorted([x.decode() for x in classlist]))
    }
    with open(os.path.join(dataset, "gt.json"), "w") as fp:
        json.dump({
            "class-list": dict_class,
            "database": data
        },
                  fp,
                  indent=4,
                  sort_keys=True)


if __name__ == "__main__":
    subsample_anet()
    create_gt()
