import torch
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import deepchem as dc
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch.utils.data as Data



class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='GDSC', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            features, edges_idx = data_mol[0], data_mol[1]
            GCNDATA = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edges_idx))
            data_list.append(GCNDATA)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA

def calculate_feat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat,adj_index]

def feature_ext(drug_feature):
    drug_data = [[] for item in range(len(drug_feature))]
    for i in range(len(drug_feature)):
        feat_mat,adj_list,_ = drug_feature.iloc[i]
        drug_data[i] = calculate_feat(feat_mat,adj_list)
    return drug_data

def cmask(num, ratio, seed):
    mask = np.ones(num, dtype=bool)
    mask[0:int(ratio * num)] = False
    np.random.seed(seed)
    np.random.shuffle(mask)
    return mask
def load_data(cn_f, exp_f, mut_f, meth_f, meta_f, prot_f, smiles_f, response_f, device):
    exp = pd.read_csv(exp_f, index_col=0)
    cn = pd.read_csv(cn_f, index_col=0)
    mut = pd.read_csv(mut_f, index_col=0)
    meta = pd.read_csv(meta_f, index_col=0)
    meth = pd.read_csv(meth_f, index_col=0)
    prot = pd.read_csv(prot_f, index_col=0).fillna(0)
    smiles = pd.read_csv(smiles_f, index_col=0)
    data = pd.read_csv(response_f, index_col=0)
    data_idx = []
    data_new = []
    data_tmp = []
    for t in zip(data['c'], data['d'], data['r']):
        data_idx.append((t[0], t[1], t[2]))

    data_sort = sorted(data_idx, key=lambda x: [x[0], x[1], x[2]], reverse=True)
    data_t = [[i[0], i[1]] for i in data_sort]
    for i,k in zip(data_t, data_sort):
        if i not in data_tmp:
            data_tmp.append(i)
            data_new.append(k)

    nb_cell = len(set([i[0] for i in data_new]))
    nb_drug = len(set([i[1] for i in data_new]))
    print('All %d pairs across %d cell lines and %d drugs.' % (len(data_new), nb_cell, nb_drug))
    triples = pd.DataFrame(data_new).to_numpy()
    triples[:, [1, 2]] = triples[:, [2, 1]]
    triples = pd.DataFrame(triples)
    triples.to_csv("data/response_triples_entity.csv", index=False)
    data = torch.tensor(np.array(data_new), dtype=torch.long)
    return smiles, exp, cn, mut, meta, meth, prot, data_new, nb_cell, nb_drug, data

def process_feat(smiles, exp, cn, mut, meta, meth, prot, data_new, nb_cell, nb_drug):
    cell_id = list(set([i[0] for i in data_new]))
    cell_id.sort()
    drug_id = list(set([i[1] for i in data_new]))
    drug_id.sort()
    cell_map = list(zip(cell_id, list(range(len(cell_id)))))
    drug_map = list(zip(drug_id, list(range(len(cell_id), len(cell_id)+len(drug_id)))))

    cell_num = np.squeeze([[j[1] for j in cell_map if i[0]==j[0]] for i in data_new])
    drug_num = np.squeeze([[j[1] for j in drug_map if i[1]==j[0]] for i in data_new])
    label_num = np.squeeze([i[2] for i in data_new])

    all_pairs = np.vstack((cell_num, drug_num, label_num)).T
    all_pairs = all_pairs[all_pairs[:, 2].argsort()]

    # drug feature
    drug_feat = {}
    feat = dc.feat.ConvMolFeaturizer()

    for id,smiles  in zip(smiles['index'], smiles['smiles']):
        if pd.isnull(id):
            continue
        mol = Chem.MolFromSmiles(smiles)
        X = feat.featurize(mol)
        drug_feat[id-192] = [X[0].get_atom_features(), X[0].get_adjacency_list(), 1]
    # t_id = list(map((lambda x:x-192), drug_id))
    drug_feat = pd.DataFrame(drug_feat).T
    drug_feat = drug_feat.loc[drug_id]
    atom_shape = drug_feat[0][0].shape[-1]
    drug_data = feature_ext(drug_feat)

    # cell feature
    exp_feat = exp.values.astype(np.float32)
    mut_feat = mut.values.astype(np.float32)
    cn_feat = cn.values.astype(np.float32)
    meta_feat = meta.values.astype(np.float32)
    meth_feat = meth.values.astype(np.float32)
    prot_feat = prot.values.astype(np.float32)
    mutf = torch.from_numpy(mut_feat)
    mutf = torch.unsqueeze(mutf, dim=1)
    mutf = torch.unsqueeze(mutf, dim=1)
    expf = torch.from_numpy(exp_feat)
    expf = torch.unsqueeze(expf, dim=1)
    expf = torch.unsqueeze(expf, dim=1)
    # expf = torch.from_numpy(exp_feat)
    cnf = torch.from_numpy(cn_feat)
    cnf = torch.unsqueeze(cnf, dim=1)
    cnf = torch.unsqueeze(cnf, dim=1)
    metaf = torch.from_numpy(meta_feat)
    metaf = torch.unsqueeze(metaf, dim=1)
    metaf = torch.unsqueeze(metaf, dim=1)
    methf = torch.from_numpy(meth_feat)
    methf = torch.unsqueeze(methf, dim=1)
    methf = torch.unsqueeze(methf, dim=1)
    protf = torch.from_numpy(prot_feat)
    protf = torch.unsqueeze(protf, dim=1)
    protf = torch.unsqueeze(protf, dim=1)

    cell_dim = [expf.shape[-1], cnf.shape[-1], metaf.shape[-1], methf.shape[-1], protf.shape[-1]]



    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_data), collate_fn=collate, batch_size=nb_drug,
                               shuffle=False)
    cell_set = Data.DataLoader(dataset=Data.TensorDataset(mutf, expf, cnf, metaf, methf, protf),
                                   batch_size=nb_cell,shuffle=False)

    edge_mask = cmask(len(all_pairs), 0.1, 666)

    train_edge = all_pairs[edge_mask]

    train_edge = np.vstack((train_edge, train_edge[:, [1, 0, 2]]))

    for i, (drug, cell) in enumerate(zip(drug_set, cell_set)):
        drug_feature = drug
        cell_feature = cell

    return train_edge, atom_shape, cell_set, drug_feature, cell_dim







