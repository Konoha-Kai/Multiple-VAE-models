import scanpy
import pandas as pd
import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# from data_process import use_h5ad
import bulk2space
from bulk2space import Bulk2Space
model = Bulk2Space()
# input_file = "data_process/output.h5ad"
    
#     # 读取部分数据
# print("\n读取部分数据用于预览")
#cell_info, expr_matrix, adata = use_h5ad.read_h5ad_data(input_file, sample_n=1000)

# # 可用
# generate_sc_meta, generate_sc_data = model.train_vae_and_generate(
#     input_bulk_path='tutorial/data/example_data/demo1/demo1_bulk.csv',
#     input_sc_data_path='tutorial/data/example_data/demo1/demo1_sc_data.csv',
#     input_sc_meta_path='tutorial/data/example_data/demo1/demo1_sc_meta.csv',
#     input_st_data_path='tutorial/data/example_data/demo1/demo1_st_data.csv',
#     input_st_meta_path='tutorial/data/example_data/demo1/demo1_st_meta.csv',
#     ratio_num=1,
#     top_marker_num=500,
#     gpu=0,
#     batch_size=512,
#     learning_rate=1e-4,
#     hidden_size=256,
#     epoch_num=100,
#     vae_save_dir='tutorial/data/example_data/demo1/predata/save_model',
#     vae_save_name='demo1_vae',
#     generate_save_dir='tutorial/data/example_data/demo1/predata/output',
#     generate_save_name='demo1'
# )


# # 可用
# generate_sc_meta, generate_sc_data = model.train_multiple_vae_and_generate(
#     input_bulk_path='tutorial/data/example_data/demo1/demo1_bulk.csv',
#     input_sc_data_path='tutorial/data/example_data/demo1/demo1_sc_data.csv',
#     input_sc_meta_path='tutorial/data/example_data/demo1/demo1_sc_meta.csv',
#     input_st_data_path='tutorial/data/example_data/demo1/demo1_st_data.csv',
#     input_st_meta_path='tutorial/data/example_data/demo1/demo1_st_meta.csv',
#     ratio_num=1,
#     top_marker_num=500,
#     gpu=0,
#     batch_size=512,
#     learning_rate=1e-4,
#     hidden_size=256,
#     epoch_num=100,
#     vae_save_dir='tutorial/data/example_data/demo1/predata/save_model',
#     vae_save_name='demo1_vae',
#     generate_save_dir='tutorial/data/example_data/demo1/predata/output',
#     generate_save_name='demo1')


# 还在搞
# generate_sc_meta, generate_sc_data = model.train_vae_and_generate_h5ad(
#     input_bulk_path='tutorial/data/example_data/demo1/demo1_bulk.csv',
#     input_sc_h5ad_path = '/root/autodl-tmp/bulk2space_new_year/data_process/output.h5ad',
#     input_st_data_path='tutorial/data/example_data/demo1/demo1_st_data.csv',
#     input_st_meta_path='tutorial/data/example_data/demo1/demo1_st_meta.csv',
#     ratio_num=1,
#     top_marker_num=500,
#     gpu=0,
#     batch_size=512,
#     learning_rate=1e-4,
#     hidden_size=256,
#     epoch_num=100,
#     vae_save_dir='tutorial/data/example_data/demo1/predata/save_model',
#     vae_save_name='demo1_vae',
#     generate_save_dir='tutorial/data/example_data/demo1/predata/output',
#     generate_save_name='demo1')