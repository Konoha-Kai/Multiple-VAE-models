import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.optimize import nnls
import scanpy as sc
from scipy.sparse import issparse
def read_h5ad_data(file_path, sample_n=None):
    """
    从h5ad文件中读取数据，并使用原始 UMI 计数进行 TPM 标准化
    
    参数:
    file_path (str): h5ad 文件路径
    sample_n (int, optional): 如果指定，随机抽取的细胞数量
    
    返回:
    tuple: (cell_info_df, expression_df, adata)
    """
    print("读取 h5ad 文件...")
    adata = sc.read_h5ad(file_path)
    
    # 如果需要抽样
    if sample_n is not None and sample_n < adata.n_obs:
        np.random.seed(42)
        selected_cells = np.random.choice(adata.obs_names, sample_n, replace=False)
        adata = adata[selected_cells]
    
    # 提取细胞信息 
    print("提取细胞信息...")
    cell_info = pd.DataFrame({
        'Cell': adata.obs_names,
        'Cell_type': adata.obs['Cell_Type'] if 'Cell_Type' in adata.obs else 'Unknown'
    })
    
    # 使用 raw.X 作为表达矩阵
    print("提取原始表达矩阵 (UMI counts)...")
    if adata.raw is not None and adata.raw.X is not None:
        expr_matrix = adata.raw.X  # 使用原始 UMI 计数
        if issparse(expr_matrix):
            expr_matrix = expr_matrix.toarray()  # 转换为密集矩阵
        expr_matrix = pd.DataFrame(expr_matrix, index=adata.obs_names, columns=adata.raw.var_names)
    else:
        print("警告: adata.raw 不存在，使用当前 adata.X 作为表达矩阵！")
        expr_matrix = adata.to_df()
    
    # 进行 TPM 标准化
    print("进行 TPM 标准化...")
    if 'feature_length' in adata.var.columns:
        # 获取基因长度 (仍然使用 var，因为 raw.var 可能不包含 feature_length)
        gene_lengths = adata.var['feature_length'].reindex(expr_matrix.columns)
        
        # 计算 RPK (Reads Per Kilobase)
        rpk = expr_matrix.div(gene_lengths, axis=1)
        
        # 计算 TPM
        scale_factor = rpk.sum(axis=1) / 1e6
        normalized_matrix = rpk.div(scale_factor, axis=0)
    else:
        print("警告：未找到基因长度信息，使用简化的标准化方法...")
        scale_factor = expr_matrix.sum(axis=1) / 1e6
        normalized_matrix = expr_matrix.div(scale_factor, axis=0)
    
    # 转置矩阵，使细胞名为列名，基因名为行名
    normalized_matrix = normalized_matrix.T
    
    print("处理完成！")
    
    # 打印保留的数据信息
    print("\n数据信息概览:")
    print(f"细胞数量: {adata.n_obs}")
    print(f"基因数量: {adata.n_vars}")
    print(f"基因矩阵的前5行: \n{normalized_matrix.head()}")
    print(f"obs 中的列: {list(adata.obs.columns)}")
    print(f"var 中的列: {list(adata.var.columns)}")
    if hasattr(adata, 'uns'):
        print(f"uns 中的键: {list(adata.uns.keys())}")
    if hasattr(adata, 'obsm'):
        print(f"obsm 中的键: {list(adata.obsm.keys())}")
    
    return cell_info, normalized_matrix, adata


def load_data(input_bulk_path,
              input_sc_data_path,
              input_sc_meta_path,
              input_st_data_path,
              input_st_meta_path):
    input_sc_meta_path = input_sc_meta_path
    input_sc_data_path = input_sc_data_path
    input_bulk_path = input_bulk_path
    input_st_meta_path = input_st_meta_path
    input_st_data_path = input_st_data_path
    print("loading data......")
    input_data = {}
    # load sc_meta.csv file, containing two columns of cell name and cell type
    input_data["input_sc_meta"] = pd.read_csv(input_sc_meta_path, index_col=0)
    # load sc_data.csv file, containing gene expression of each cell
    input_sc_data = pd.read_csv(input_sc_data_path, index_col=0)
    input_data["sc_gene"] = input_sc_data._stat_axis.values.tolist()
    # load bulk.csv file, containing one column of gene expression in bulk
    input_bulk = pd.read_csv(input_bulk_path, index_col=0)
    input_data["bulk_gene"] = input_bulk._stat_axis.values.tolist()
    # filter overlapping genes.
    input_data["intersect_gene"] = list(set(input_data["sc_gene"]).intersection(set(input_data["bulk_gene"])))
    input_data["input_sc_data"] = input_sc_data.loc[input_data["intersect_gene"]]
    input_data["input_bulk"] = input_bulk.loc[input_data["intersect_gene"]]
    # load st_meta.csv and st_data.csv, containing coordinates and gene expression of each spot respectively.
    input_data["input_st_meta"] = pd.read_csv(input_st_meta_path, index_col=0)
    input_data["input_st_data"] = pd.read_csv(input_st_data_path, index_col=0)
    print("load data done!")
    
    return input_data
def load_h5ad(input_bulk_path,
              input_sc_h5ad_path,
              input_st_data_path,
              input_st_meta_path):
    sc_meta_df, sc_data_df, adata = read_h5ad_data(input_sc_h5ad_path, sample_n=1000)
    input_bulk_path = input_bulk_path
    input_st_meta_path = input_st_meta_path
    input_st_data_path = input_st_data_path
    print("loading data......")
    input_data = {}
    # load sc_meta.csv file, containing two columns of cell name and cell type
    input_data["input_sc_meta"] = sc_meta_df
    # load sc_data.csv file, containing gene expression of each cell
    input_sc_data = sc_data_df
    input_data["sc_gene"] = input_sc_data._stat_axis.values.tolist()
    # load bulk.csv file, containing one column of gene expression in bulk
    input_bulk = pd.read_csv(input_bulk_path, index_col=0)
    input_data["bulk_gene"] = input_bulk._stat_axis.values.tolist()
    # filter overlapping genes.
    input_data["intersect_gene"] = list(set(input_data["sc_gene"]).intersection(set(input_data["bulk_gene"])))
    input_data["input_sc_data"] = input_sc_data.loc[input_data["intersect_gene"]]
    input_data["input_bulk"] = input_bulk.loc[input_data["intersect_gene"]]
    # load st_meta.csv and st_data.csv, containing coordinates and gene expression of each spot respectively.
    input_data["input_st_meta"] = pd.read_csv(input_st_meta_path, index_col=0)
    input_data["input_st_data"] = pd.read_csv(input_st_data_path, index_col=0)
    print("load data done!")
    
    return input_data
def data_process(data, a, ratio_num):
    # Data processing
    breed = data["input_sc_meta"]['Cell_type']
    breed_np = breed.values
    breed_set = set(breed_np)
    id2label = sorted(list(breed_set))  # List of breed
    label2id = {label: idx for idx, label in enumerate(id2label)}  # map breed to breed-id

    cell2label = dict()  # map cell-name to breed-id
    label2cell = defaultdict(set)  # map breed-id to cell-names
    for row in data["input_sc_meta"].itertuples():
        cell_name = row.Index
        cell_type = label2id[row.Cell_type]
        cell2label[cell_name] = cell_type
        label2cell[cell_type].add(cell_name)

    label_devide_data = dict()
    for label, cells in label2cell.items():
        # print('list(cells)',list(cells))
        label_devide_data[label] = data["input_sc_data"][list(cells)]

    single_cell_splitby_breed_np = {}
    for key in label_devide_data.keys():
        single_cell_splitby_breed_np[key] = label_devide_data[key].values  # [gene_num, cell_num]
        single_cell_splitby_breed_np[key] = single_cell_splitby_breed_np[key].mean(axis=1)

    max_decade = len(single_cell_splitby_breed_np.keys())
    single_cell_matrix = []

    for i in range(max_decade):
        single_cell_matrix.append(single_cell_splitby_breed_np[i].tolist())

    single_cell_matrix = np.array(single_cell_matrix)
    single_cell_matrix = np.transpose(single_cell_matrix)  # (gene_num, label_num)

    bulk_marker = data["input_bulk"].values  # (gene_num, 1)
    bulk_rep = bulk_marker.reshape(bulk_marker.shape[0], )

    # calculate celltype ratio in each spot by NNLS
    ratio = nnls(single_cell_matrix, bulk_rep)[0]
    ratio = ratio / sum(ratio)

    ratio_array = np.round(ratio * data["input_sc_meta"].shape[0] * ratio_num)
    ratio_list = [r for r in ratio_array]

    cell_target_num = dict(zip(id2label, ratio_list))

    return cell_target_num

# 示例调用

