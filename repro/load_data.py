import dgl
import torch as th
import numpy as np
import pandas as pd


def load(dataset):
    return load_Kdataset()


def load_Kdataset():
    """建 Kdataset 的異質圖。894 藥 × 504 病(454 人類 + 50 寵物)。

    寵物疾病已併入原始檔案,沒有獨立的 KPet 資料。建圖方式與上游相同,
    唯一的差別是 disease_sim 要補零 —— disease_disease_baseline.csv 只涵蓋
    人類病(454×454),寵物病沒有 MeSH 相似度,特徵為 0,靠橋接邊傳訊息。
    """
    KD = './dataset/Kdataset'
    drug_drug = pd.read_csv(KD + '/drug_drug_baseline.csv', header=None).values
    drug_sim = drug_drug
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    protein_protein = pd.read_csv(KD + '/interactions/protein_protein.csv')
    gene_gene = pd.read_csv(KD + '/interactions/gene_gene.csv')
    pathway_pathway = pd.read_csv(KD + '/interactions/pathway_pathway.csv')
    disease_disease = pd.read_csv(KD + '/disease_disease_baseline.csv', header=None).values
    n_human = disease_disease.shape[0]
    disease_sim_h = disease_disease
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    # 補資料:寵物 disease_disease 邊(pet<->人類同病),接上人類監督訊號。
    # 這些橋接存在 interactions/disease_disease.csv(原始檔,格式相同),取 index>=n_human 的列。
    # 註:disease_disease_baseline.csv 維持 454×454 —— 它被上面的 top-15 邏輯消費,
    #     寵物列若放進去會憑空多 15 條相似邊(實測 100→2648),破壞「寵物病只有 1 條橋接」的設計。
    # 只取 Sim==1.0 的當【邊】—— 那是「寵物病↔同名人類病」的橋接(96 條)。
    # 檔案裡另有寵物病對其他人類病的 Wang 相似度(Sim<1),那是給 run_all.py 的
    # prop 用的【權重】,不是圖上的邊;若一併當邊會讓寵物邊從 96 暴增到 4498,
    # 大幅改變 GNN 的圖結構。
    _dd_all = pd.read_csv(KD + '/interactions/disease_disease.csv')
    _pet_rows = (_dd_all['Disease1'] >= n_human) | (_dd_all['Disease2'] >= n_human)
    pet_dd = _dd_all[_pet_rows & (_dd_all['Sim'] == 1.0)][['Disease1', 'Disease2']]
    disease_disease = pd.concat([disease_disease, pet_dd], ignore_index=True)
    drug_protein = pd.read_csv(KD + '/associations/drug_protein.csv')
    protein_gene = pd.read_csv(KD + '/associations/protein_gene.csv')
    gene_pathway = pd.read_csv(KD + '/associations/gene_pathway.csv')
    pathway_disease = pd.read_csv(KD + '/associations/pathway_disease.csv')
    # 藥-病關聯:已含寵物標籤(associations/Kdataset.csv 的 Evidence 欄此處不讀)
    drug_disease = pd.read_csv(KD + '/associations/Kdataset.csv')[['Drug', 'Disease']]
    # 疾病總數 = omics/disease.csv 的列數(人類 + 寵物都在裡面)
    n_dis = len(pd.read_csv(KD + '/omics/disease.csv'))
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),
        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        ('protein', 'protein_gene', 'gene'): (th.tensor(protein_gene['Protein'].values),
                                              th.tensor(protein_gene['Gene'].values)),
        ('gene', 'gene_protein', 'protein'): (th.tensor(protein_gene['Gene'].values),
                                              th.tensor(protein_gene['Protein'].values)),
        ('gene', 'gene_gene', 'gene'): (th.tensor(gene_gene['Gene1'].values),
                                        th.tensor(gene_gene['Gene2'].values)),
        ('gene', 'gene_pathway', 'pathway'): (th.tensor(gene_pathway['Gene'].values),
                                              th.tensor(gene_pathway['Pathway'].values)),
        ('pathway', 'pathway_gene', 'gene'): (th.tensor(gene_pathway['Pathway'].values),
                                              th.tensor(gene_pathway['Gene'].values)),
        ('pathway', 'pathway_pathway', 'pathway'): (th.tensor(pathway_pathway['Pathway1'].values),
                                                    th.tensor(pathway_pathway['Pathway2'].values)),
        ('pathway', 'pathway_disease', 'disease'): (th.tensor(pathway_disease['Pathway'].values),
                                                    th.tensor(pathway_disease['Disease'].values)),
        ('disease', 'disease_pathway', 'pathway'): (th.tensor(pathway_disease['Disease'].values),
                                                    th.tensor(pathway_disease['Pathway'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), n_dis))))
    disease_sim = np.zeros((n_dis, n_dis))
    disease_sim[:n_human, :n_human] = disease_sim_h     # 寵物疾病無相似度特徵(=0),靠 pathway 訊息傳遞
    dis_feature = np.hstack((np.zeros((n_dis, g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.zeros((g.num_nodes('protein'), drug_feature.shape[1])).to(th.float32)
    g.nodes['gene'].data['h'] = th.zeros((g.num_nodes('gene'), drug_feature.shape[1])).to(th.float32)
    g.nodes['pathway'].data['h'] = th.zeros((g.num_nodes('pathway'), drug_feature.shape[1])).to(th.float32)
    return g


def remove_graph(g, test_id):
    """Delete the drug-disease associations which belong to test set
    from heterogeneous network.
    """

    test_drug_id = test_id[:, 0]
    test_dis_id = test_id[:, 1]
    edges_id = g.edge_ids(th.tensor(test_drug_id),
                          th.tensor(test_dis_id),
                          etype=('drug', 'drug_disease', 'disease'))
    g = dgl.remove_edges(g, edges_id, etype=('drug', 'drug_disease', 'disease'))
    edges_id = g.edge_ids(th.tensor(test_dis_id),
                          th.tensor(test_drug_id),
                          etype=('disease', 'disease_drug', 'drug'))
    g = dgl.remove_edges(g, edges_id, etype=('disease', 'disease_drug', 'drug'))
    return g
