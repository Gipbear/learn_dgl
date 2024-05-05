import os

import dgl
import torch
import pandas as pd
import numpy as np
from dgl.data import DGLBuiltinDataset


class ArixvDataSet(DGLBuiltinDataset):
    def __init__(self, name, url=None, raw_dir=None, hash_key=..., force_reload=False, verbose=False, transform=None):
        assert name.lower() == 'arixv'
        super().__init__(name, url, raw_dir, hash_key, force_reload, verbose, transform)

    def process(self):
        edges_path = os.path.join(self.raw_path, 'edges.csv')
        feat_path = os.path.join(self.raw_path, 'feat.npy')
        train_path = os.path.join(self.raw_path, 'train.csv')
        test_path = os.path.join(self.raw_path, 'test.csv')

        # node and edge
        edges_data = pd.read_csv(edges_path, header=None, names=['src', 'dst']).values
        edges_src = torch.from_numpy(edges_data[:, 0])
        edges_dst = torch.from_numpy(edges_data[:, 1])
        node_feat = np.load(feat_path)
        n_nodes = node_feat.shape[0]
        self._g = dgl.graph((edges_src, edges_dst), num_nodes=n_nodes)
        self._g.ndata['feat'] = torch.from_numpy(node_feat)
        # node label
        train_data = pd.read_csv(train_path, header=0, names=['nid', 'label'], dtype=int).values
        label = -np.ones(n_nodes, dtype=int)
        label[train_data[:, 0]] = train_data[:, 1]
        self._g.ndata['label'] = torch.from_numpy(label)
        self.labels = self._g.ndata['label'].unsqueeze(dim=1)
        self._g.num_classes = len(torch.unique(self.labels)) - 1
        # mask
        train_part = int(train_data.shape[0] * 0.8)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[train_data[:train_part, 0]] = True
        val_mask[train_data[train_part:, 0]] = True
        test_data = pd.read_csv(test_path, header=0).values
        test_mask[test_data] = True
        self._g.ndata['train_mask'] = train_mask
        self._g.ndata['val_mask'] = val_mask
        self._g.ndata['test_mask'] = test_mask

    def get_idx_split(self):
        train_idx = torch.nonzero(self._g.ndata['train_mask']).squeeze(dim=1).to(torch.long)
        valid_idx = torch.nonzero(self._g.ndata['val_mask']).squeeze(dim=1).to(torch.long)
        test_idx = torch.nonzero(self._g.ndata['test_mask']).squeeze(dim=1).to(torch.long)
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._g, self.labels
        else:
            return self._transform(self._g), self.labels

    def __len__(self):
        return 1


if __name__ == '__main__':
    data_set = ArixvDataSet('arixv', raw_dir='data')
    graph, labels = data_set[0]
    print(labels.max())
