import argparse
import os
import os.path as osp
import pickle
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', help='CSL-Daily sentence_label dir (csl2020ct_v2.pkl, split_1.txt)', required=True)
    parser.add_argument('--frame_root', help='location of CSL-Daily_256x256px', required=True)
    parser.add_argument('--out_dir', help='where to save the output', default='./preprocess/CSL-Daily')
    return parser


def load_split(label_dir):
    split = {'train': [], 'dev': [], 'test': []}
    with open(osp.join(label_dir, 'split_1.txt')) as f:
        next(f)  # header: name|split
        for line in f:
            line = line.strip()
            if line:
                name, mode = line.split('|')
                split[mode].append(name)
    return split


def main():
    parser = get_parser()
    args = parser.parse_args()

    with open(osp.join(args.label_dir, 'csl2020ct_v2.pkl'), 'rb') as f:
        info = {d['name']: d for d in pickle.load(f)['info']}
    split = load_split(args.label_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    for mode, names in split.items():
        data = {}
        for name in names:
            if name not in info:
                print(f"Warning: {name} ({mode}) has no annotation, skipped.")
                continue
            if not osp.isdir(osp.join(args.frame_root, mode, name)):
                print(f"Warning: {name} ({mode}) has no frame folder, skipped.")
                continue

            d = info[name]
            data[len(data)] = {
                'fileid': name,
                'folder': f'{mode}/{name}/*.jpg',
                'text': ''.join(d['label_char']),
                'gloss': ' '.join(d['label_gloss']),
                'signer': d['signer'],
                'num_frames': d['length'],
            }
        data['prefix'] = args.frame_root

        # No cross-lingual translations exist for CSL-Daily, so _ml is identical
        for fname in [f'{mode}_info.npy', f'{mode}_info_ml.npy']:
            np.save(osp.join(args.out_dir, fname), data)
        print(f"{mode}: {len(data) - 1} samples")


if __name__ == "__main__":
    main()
