import argparse
import tqdm
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from model import *
from dataset import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def train(cn_f, exp_f, mut_f, meth_f, meta_f, prot_f, smiles_f, response_f, epochs):
    smiles, exp, cn, mut, meta, meth, prot, data_new, nb_cell, nb_drug, data = load_data(cn_f, exp_f, mut_f, meth_f, meta_f,
                                                                                   prot_f, smiles_f, response_f, device)
    edge, atom_shape, cell_feature, drug_feature, cell_dim = process_feat(smiles, exp, cn, mut, meta, meth, prot, data_new, nb_cell, nb_drug)
    label = data[:, 2:3]
    idx = np.arange(data.shape[0])
    train_idx, tst_idx, train, tst = train_test_split(idx, data, test_size=0.2, random_state=42, shuffle=True, stratify=label)
    trn_idx, val_idx, trn, val = train_test_split(np.arange(train.shape[0]), train, test_size=0.2, random_state=42, shuffle=True, stratify=label[train_idx])
    drug_feature.to(device)
    data = data.to(device)
    label = torch.tensor(label).to(device)

    model = DRP(atom_shape, device, layersnums=layer_num, hidden_size=hidden_size, att_heads=att_head, dropout=dropout,
                emb_output=output, out_size=out_size, feat_dim_li=cell_dim).to(device)
    optim = torch.optim.AdamW(lr=ptlr, weight_decay=ptwd, params=model.parameters())
    l = nn.CrossEntropyLoss()
    early = EarlyStopping(patience=3, verbose=True)
    b_acc = 0
    b_auc = 0
    b_aupr = 0
    for e in tqdm.tqdm(range(epochs)):
        model.train()
        out, loss, _ = model(data, drug_feature.x, drug_feature.edge_index, drug_feature.batch, cell_feature, edge, nb_cell)
        out = out[trn_idx]
        optim.zero_grad()
        loss = l(out, label[trn_idx].reshape(-1)) + loss
        loss.backward()
        optim.step()

        trn_acc = (out.argmax(dim=1) == label[trn_idx].reshape(-1)).sum(dtype=float) / len(trn_idx)
        trn_auc = get_auc(out, label[trn_idx])
        trn_aupr = get_aupr(out, label[trn_idx])
        precision, recall, f1 = get_confusion(out, label[trn_idx])

        if e % 1 == 0:
            print("train acc={:.4f}|train auc={:.4f}|train aupr={:.4f}|train precision={:.4f}|train recall={:.4f}|train f1={:.4f}".format(
                trn_acc.item(), trn_auc.item(), trn_aupr.item(), precision.item(), recall.item(), f1.item()))
        model.eval()

        with torch.no_grad():
            out, val_loss, _ = model(data, drug_feature.x, drug_feature.edge_index, drug_feature.batch, cell_feature, edge,
                              nb_cell)
            out = out[val_idx]
            val_loss = l(out, label[val_idx].reshape(-1)) + val_loss

            val_acc = (out.argmax(dim=1) == label[val_idx].reshape(-1)).sum(dtype=float) / len(val_idx)
            val_auc = get_auc(out, label[val_idx])
            val_aupr = get_aupr(out, label[val_idx])
            val_p, val_r, val_f1 = get_confusion(out, label[val_idx])
            if e % 1 == 0:
                print("val acc={:.4f}|val auc={:.4f}|val aupr={:.4f}|val precision={:.4f}|val recall={:.4f}|val f1={:.4f}".format(
                    val_acc.item(), val_auc.item(), val_aupr.item(), val_p, val_r, val_f1))
            b_acc = max(b_acc, val_acc)
            b_auc = max(b_auc, val_auc)
            b_aupr = max(b_aupr, val_aupr)
            early(val_loss, model)
            if early.early_stop:
                print("Early stopping, best epoch: {:d}".format(e + 1))
                break
    print("best epoch{:d}|best acc={:.4f}|best auc={:.4f}|best aupr={:.4f}".format(e+1, b_acc.item(), b_auc.item(), b_aupr.item()))
    predict(model, data, drug_feature, cell_feature, edge, nb_cell, tst_idx, label)


def predict(model, data, drug_feature, cell_feature, edge, nb_cell, tst_idx, label):
    with torch.no_grad():
        out, _, emb = model(data, drug_feature.x, drug_feature.edge_index, drug_feature.batch, cell_feature, edge,
                              nb_cell)

        out = out[tst_idx]

        tst_acc = (out.argmax(dim=1) == label[tst_idx].reshape(-1)).sum(dtype=float) / len(tst_idx)
        tst_auc = get_auc(out, label[tst_idx])
        tst_aupr = get_aupr(out, label[tst_idx])
        precision, recall, f1 = get_confusion(out, label[tst_idx])
        print(
            'acc={:.4f}|auc={:.4f}|aupr={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}'.format(
                tst_acc, tst_auc, tst_aupr, precision, recall, f1))




if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    set_random_seed(42)
    args = argparse.ArgumentParser()

    args.add_argument("--learning_rate", default=0.001)
    args.add_argument("--hidden_size", default=128)
    args.add_argument("--out_size", default=128)
    args.add_argument("--dropout", default=0.2)
    args.add_argument("--ptwd", default=0)
    args.add_argument("--epochs", default=100)
    args.add_argument("--output", default=128)
    args.add_argument("--att_head", default=3)
    args.add_argument("--layer_num", default=2)

    args = args.parse_args()

    hidden_size = args.hidden_size
    out_size = args.out_size
    dropout = args.dropout
    ptlr = args.learning_rate
    ptwd = args.ptwd
    epochs = args.epochs
    output = args.output
    att_head = args.att_head
    layer_num = args.layer_num

    exp_f = 'data/cell_line_expression.csv'
    cn_f = 'data/cell_line_copynumber.csv'
    mut_f = 'data/cell_line_mutation.csv'
    meta_f = 'data/cell_line_metabolomic.csv'
    meth_f = 'data/cell_line_methylation.csv'
    prot_f = 'data/cell_line_protein.csv'
    smiles_f = 'data/drug.csv'
    response_f = 'data/response_bak.csv'
    train(cn_f, exp_f, mut_f, meth_f, meta_f, prot_f, smiles_f, response_f, epochs)
