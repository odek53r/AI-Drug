"""KPet 折平行訓練 + loss on GPU。

與原 main.py 的差異只有兩點(均不影響演算法/準度):
  ① 每折獨立成行程,平行塞進閒置 GPU(folds 本就互相獨立)
  ② loss 在 GPU 上算(原本搬回 CPU),數學完全相同
每折的訓練/測試流程、optimizer、scheduler、criterion、epoch、早停、mask 與 main.py 一致。
model.py 演算法完全未動。

用法:
  python train_parallel.py -da KPet --mode fold --fold k -sp resultKPet_par   # 訓練+測試第 k 折
  python train_parallel.py -da KPet --mode aggregate     -sp resultKPet_par   # 彙整成 result.csv + 指標
"""
import os
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from model import Model
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from load_data import load, remove_graph
from utils import get_metrics_auc, set_seed, m2v, plot_result_auc, plot_result_aupr, EarlyStopping, get_metrics
from args import args


def prepare():
    simplefilter(action='ignore', category=FutureWarning)
    df = pd.read_csv('./dataset/{}/{}_baseline.csv'.format(args.dataset, args.dataset), header=None).values
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])]).astype('int64')
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)
    splits = list(zip(kf.split(data_pos), kf.split(data_neg)))
    return df, data, data_pos, data_neg, splits


def _feature_metapath(g):
    feature = {'drug': g.nodes['drug'].data['h'], 'disease': g.nodes['disease'].data['h'],
               'protein': g.nodes['protein'].data['h'], 'gene': g.nodes['gene'].data['h'],
               'pathway': g.nodes['pathway'].data['h']}
    metapath = ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease']
    return feature, metapath


def run_one_fold(fold):
    device = th.device('cuda:{}'.format(args.device_id)) if args.device_id else th.device('cpu')
    set_seed(args.seed)                       # 每折以相同種子初始化(確定且可重現)
    df, data, data_pos, data_neg, splits = prepare()
    (tr_pos, te_pos), (tr_neg, te_neg) = splits[fold]
    train_pos_id, test_pos_id = data_pos[tr_pos], data_pos[te_pos]
    train_neg_id, test_neg_id = data_neg[tr_neg], data_neg[te_neg]
    train_pos_idx = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
    test_pos_idx = [tuple(test_pos_id[:, 0]), tuple(test_pos_id[:, 1])]
    train_neg_idx = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]
    test_neg_idx = [tuple(test_neg_id[:, 0]), tuple(test_neg_id[:, 1])]

    g = load(args.dataset)
    g = remove_graph(g, test_pos_id[:, :-1]).to(device)
    feature, metapath = _feature_metapath(g)

    mask_label = np.ones(df.shape)
    mask_label[test_pos_idx[0], test_pos_idx[1]] = 0
    mask_label[test_neg_idx[0], test_neg_idx[1]] = 0
    mask_test = np.where(mask_label == 0); mask_test = [tuple(mask_test[0]), tuple(mask_test[1])]
    mask_train = np.where(mask_label == 1); mask_train = [tuple(mask_train[0]), tuple(mask_train[1])]
    label = th.tensor(df).float().to(device)

    drug_emb, disease_emb = m2v(g, metapath)
    model = Model(etypes=g.etypes, ntypes=g.ntypes, in_feats=feature['drug'].shape[1],
                  hidden_feats=args.hidden_feats, num_heads=args.num_heads, dropout=args.dropout).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optim_scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * args.learning_rate,
                                                     max_lr=args.learning_rate, gamma=0.995,
                                                     step_size_up=20, mode="exp_range", cycle_momentum=False)
    pw = len(train_neg_idx[0]) / len(train_pos_idx[0])
    criterion = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(pw, device=device))   # ② pos_weight 與 loss 都在 GPU
    stopper = EarlyStopping(patience=args.patience, saved_path=args.saved_path)
    stopper.filename = os.path.join(args.saved_path, 'fold_{}.pth'.format(fold))    # ① 固定檔名對應折號

    for epoch in range(1, args.epoch + 1):
        model.train()
        score = model(g, feature, drug_emb, disease_emb)
        pred = th.sigmoid(score)
        loss = criterion(score[mask_train].flatten(), label[mask_train].flatten())  # ② GPU 上算 loss
        optimizer.zero_grad(); loss.backward(); optimizer.step(); optim_scheduler.step()
        model.eval()
        AUC, _ = get_metrics_auc(label[mask_train].cpu().detach().numpy(), pred[mask_train].cpu().detach().numpy())
        early_stop = stopper.step(loss.item(), AUC, model)
        if epoch % 100 == 0:
            print('[fold {}] Epoch {} Loss {:.3f} Train AUC {:.3f}'.format(fold, epoch, loss.item(), AUC), flush=True)
            if early_stop:
                break

    # 測試本折(載入本折最佳 checkpoint)
    model.load_state_dict(th.load(stopper.filename))
    model.eval()
    pred = th.sigmoid(model(g, feature, drug_emb, disease_emb))
    AUC, AUPR = get_metrics_auc(label[mask_test].cpu().detach().numpy(), pred[mask_test].cpu().detach().numpy())
    pred = pred.cpu().detach().numpy()
    np.save(os.path.join(args.saved_path, 'pred_full_{}.npy'.format(fold)), pred)  # 整張矩陣(供 2 折快速機制驗證)
    out = np.full(df.shape, np.nan)
    out[test_pos_idx[0], test_pos_idx[1]] = pred[test_pos_idx[0], test_pos_idx[1]]
    out[test_neg_idx[0], test_neg_idx[1]] = pred[test_neg_idx[0], test_neg_idx[1]]
    np.save(os.path.join(args.saved_path, 'pred_fold_{}.npy'.format(fold)), out)
    print('[fold {}] Test AUC {:.3f}; AUPR: {:.3f}'.format(fold, AUC, AUPR), flush=True)


def run_full():
    """全資料訓練:不切折、不移除任何邊、不遮任何標籤 → 預測整張矩陣 → result_full.csv

    用途:**產候選(production)**——要預測的是「還沒有藥-病關聯」的空格,
          用光所有已知標籤才最準。
    ⚠️ 禁忌:此檔看過全部標籤,**不可**用來報 recall / AUC(那會是洩漏)。
             報效能請用 10 折 OOF 的 result.csv。
    演算法未改:model.py 零修改,這裡只是不做 fold/mask 的訓練 harness。
    """
    device = th.device('cuda:{}'.format(args.device_id)) if args.device_id else th.device('cpu')
    set_seed(args.seed)
    df, data, data_pos, data_neg, _ = prepare()

    # 全部格子都是訓練格(無測試集)。
    # ⚡ 效能:因為 mask = 全部格子,score[mask].flatten() 與 score.flatten() 逐位元相同,
    #    直接 flatten 可省掉「每 epoch 把 45 萬元素 Python tuple 轉張量索引」(實測 313ms x4/epoch)。
    #    數學完全等價,只是快 ~8 倍。
    n_pos, n_neg = len(data_pos), len(data_neg)
    label_np = df.flatten().astype(float)      # 標籤不變 → 迴圈外算一次,供 AUC 用

    g = load(args.dataset).to(device)          # ← 不 remove_graph:保留所有已知邊
    feature, metapath = _feature_metapath(g)
    label = th.tensor(df).float().to(device)

    drug_emb, disease_emb = m2v(g, metapath)
    model = Model(etypes=g.etypes, ntypes=g.ntypes, in_feats=feature['drug'].shape[1],
                  hidden_feats=args.hidden_feats, num_heads=args.num_heads, dropout=args.dropout).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optim_scheduler = th.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1 * args.learning_rate,
                                                     max_lr=args.learning_rate, gamma=0.995,
                                                     step_size_up=20, mode="exp_range", cycle_momentum=False)
    pw = n_neg / n_pos                         # 與各折同樣的正例加權(≈158)
    criterion = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(pw, device=device))
    stopper = EarlyStopping(patience=args.patience, saved_path=args.saved_path)
    stopper.filename = os.path.join(args.saved_path, 'full.pth')
    print('[full] 全資料訓練:{} 正例 / {} 負例,pos_weight={:.1f}'.format(n_pos, n_neg, pw), flush=True)

    for epoch in range(1, args.epoch + 1):
        model.train()
        score = model(g, feature, drug_emb, disease_emb)
        pred = th.sigmoid(score)
        loss = criterion(score.flatten(), label.flatten())        # ⚡ 等價於 score[全部格子]
        optimizer.zero_grad(); loss.backward(); optimizer.step(); optim_scheduler.step()
        model.eval()
        # ⚡ 只算 EarlyStopping 需要的 AUC;原本 get_metrics_auc 還會算 AUPR 但回傳後被丟棄(64ms/epoch 純浪費)
        AUC = roc_auc_score(label_np, pred.flatten().cpu().detach().numpy())
        early_stop = stopper.step(loss.item(), AUC, model)
        if epoch % 100 == 0:
            print('[full] Epoch {} Loss {:.3f} Train AUC {:.3f}'.format(epoch, loss.item(), AUC), flush=True)
            if early_stop:
                break

    model.load_state_dict(th.load(stopper.filename))
    model.eval()
    pred = th.sigmoid(model(g, feature, drug_emb, disease_emb)).cpu().detach().numpy()
    out = os.path.join(args.saved_path, 'result_full.csv')
    pd.DataFrame(pred).to_csv(out, index=False, header=False)
    print('[full] 完成 → {} (shape {})'.format(out, pred.shape), flush=True)
    print('[full] ⚠️ 只可用於產候選;報 recall/AUC 請用 10 折 OOF 的 result.csv', flush=True)


def aggregate():
    df, data, _, _, _ = prepare()
    pred_result = np.zeros(df.shape)
    for k in range(args.nfold):
        out = np.load(os.path.join(args.saved_path, 'pred_fold_{}.npy'.format(k)))
        m = ~np.isnan(out)
        pred_result[m] = out[m]
    AUC, aupr, acc, f1, pre, rec, spe = get_metrics(df.flatten().astype(float), pred_result.flatten())
    print('Overall: AUC {:.3f}; AUPR: {:.3f}; Acc: {:.3f}; F1: {:.3f}; Precision {:.3f}; Recall {:.3f}; Specificity {:.3F}'
          .format(AUC, aupr, acc, f1, pre, rec, spe))
    pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path, 'result.csv'), index=False, header=False)
    plot_result_auc(args, data[:, -1].flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, data[:, -1].flatten(), pred_result.flatten(), aupr)


if __name__ == '__main__':
    os.makedirs(args.saved_path, exist_ok=True)
    if args.mode == 'aggregate':
        aggregate()
    elif args.mode == 'full':
        run_full()
    elif args.mode == 'fold':
        run_one_fold(args.fold)
    else:
        for k in range(args.nfold):
            run_one_fold(k)
        aggregate()
