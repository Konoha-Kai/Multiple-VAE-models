import scanpy
import pandas as pd
import numpy as np
import matplotlib.colors as clr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import bulk2space
from bulk2space import Bulk2Space
model = Bulk2Space()
# multiple_vae_dict = model.train_multiple_vae_and_generate(
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
#     epoch_num=20,
#     vae_save_dir='tutorial/data/example_data/demo1/predata/save_model',
#     vae_save_name='demo1_vae',
#     generate_save_dir='tutorial/data/example_data/demo1/predata/output',
#     generate_save_name='demo1')
# print(multiple_vae_dict)
generate_sc_meta, generate_sc_data = model.train_vae_and_generate(
    input_bulk_path='tutorial/data/example_data/demo1/demo1_bulk.csv',
    input_sc_data_path='tutorial/data/example_data/demo1/demo1_sc_data.csv',
    input_sc_meta_path='tutorial/data/example_data/demo1/demo1_sc_meta.csv',
    input_st_data_path='tutorial/data/example_data/demo1/demo1_st_data.csv',
    input_st_meta_path='tutorial/data/example_data/demo1/demo1_st_meta.csv',
    ratio_num=1,
    top_marker_num=500,
    gpu=0,
    batch_size=512,
    learning_rate=1e-4,
    hidden_size=256,
    epoch_num=20,
    vae_save_dir='tutorial/data/example_data/demo1/predata/save_model',
    vae_save_name='demo1_vae',
    generate_save_dir='tutorial/data/example_data/demo1/predata/output',
    generate_save_name='demo1')
# generate_sc_meta, generate_sc_data = model.train_vae_and_generate(
#     input_bulk_path='tutorial/data/example_data/demo2/demo2_bulk.csv',
#     input_sc_data_path='tutorial/data/example_data/demo2/demo2_sc_data.csv',
#     input_sc_meta_path='tutorial/data/example_data/demo2/demo2_sc_meta.csv',
#     input_st_data_path='tutorial/data/example_data/demo2/demo2_st_data.csv',
#     input_st_meta_path='tutorial/data/example_data/demo2/demo2_st_meta.csv',
#     ratio_num=1,
#     top_marker_num=500,
#     gpu=0,
#     batch_size=512,
#     learning_rate=1e-4,
#     hidden_size=256,
#     epoch_num=1000,
#     vae_save_dir='tutorial/data/example_data/demo2/predata/save_model',
#     vae_save_name='demo2_vae',
#     generate_save_dir='tutorial/data/example_data/demo2/predata/output',
#     generate_save_name='demo2')
#
# # The number of cells per cell type in deconvoluted bulk-seq data
# ct_stat = pd.DataFrame(generate_sc_meta['Cell_type'].value_counts())
# ct_name = list(ct_stat.index)
# ct_num = list(ct_stat['Cell_type'])
# color = ["#34D399", "#F9A8D4", '#8B5CF6', "#EF4444", '#047857', '#A5B4FC', "#3B82F6", '#B45309', '#FBBF24']
# plt.bar(ct_name, ct_num, color=color)
# plt.xticks(ct_name, ct_name, rotation=90)
# plt.title("The number of cells per cell type in bulk-seq data")
# plt.xlabel("Cell type")
# plt.ylabel("Cell number")
# plt.savefig('Try1.png')
#
# # load input sc data
# input_data = bulk2space.utils.load_data(
#     input_bulk_path='tutorial/data/example_data/demo2/demo2_bulk.csv',
#     input_sc_data_path='tutorial/data/example_data/demo2/demo2_sc_data.csv',
#     input_sc_meta_path='tutorial/data/example_data/demo2/demo2_sc_meta.csv',
#     input_st_data_path='tutorial/data/example_data/demo2/demo2_st_data.csv',
#     input_st_meta_path='tutorial/data/example_data/demo2/demo2_st_meta.csv'
# )
#
# # Calculate 200 marker genes of each cell type
# sc = scanpy.AnnData(input_data['input_sc_data'].T)
# sc.obs = input_data['input_sc_meta'][['Cell_type']]
# scanpy.tl.rank_genes_groups(sc, 'Cell_type', method='wilcoxon')
# marker_df = pd.DataFrame(sc.uns['rank_genes_groups']['names']).head(200)
# marker = list(np.unique(np.ravel(np.array(marker_df))))
#
# # the mean expression of 200 marker genes of input sc data
# sc_marker = input_data['input_sc_data'].loc[marker, :].T
# sc_marker['Cell_type'] = input_data['input_sc_meta']['Cell_type']
# sc_marker_mean = sc_marker.groupby('Cell_type')[marker].mean()
#
# # the mean expression of 200 marker genes of deconvoluted bulk-seq data
# generate_sc_meta.index = list(generate_sc_meta['Cell'])
# generate_sc_data_new = generate_sc_data.T
# generate_sc_data_new['Cell_type'] = generate_sc_meta['Cell_type']
# generate_sc_marker_mean = generate_sc_data_new.groupby('Cell_type')[marker].mean()
#
# intersect_cell = list(set(sc_marker_mean.index).intersection(set(generate_sc_marker_mean.index)))
# generate_sc_marker_mean = generate_sc_marker_mean.loc[intersect_cell]
# sc_marker_mean = sc_marker_mean.loc[intersect_cell]
#
# # calculate correlation
# sc_marker_mean = sc_marker_mean.T
# generate_sc_marker_mean = generate_sc_marker_mean.T
#
# coeffmat = np.zeros((sc_marker_mean.shape[1], generate_sc_marker_mean.shape[1]))
# for i in range(sc_marker_mean.shape[1]):
#     for j in range(generate_sc_marker_mean.shape[1]):
#         corrtest = pearsonr(sc_marker_mean[sc_marker_mean.columns[i]],
#                             generate_sc_marker_mean[generate_sc_marker_mean.columns[j]])
#         coeffmat[i, j] = corrtest[0]
#
# rf_ct = list(sc_marker_mean.columns)
# generate_ct = list(generate_sc_marker_mean.columns)
#
# # plot
# fig, ax = plt.subplots()
# im = ax.imshow(coeffmat, cmap='RdBu_r')
# ax.set_xticks(np.arange(len(rf_ct)))
# ax.set_xticklabels(rf_ct)
# ax.set_yticks(np.arange(len(generate_ct)))
# ax.set_yticklabels(generate_ct)
# plt.xlabel("scRNA-seq reference")
# plt.ylabel("deconvoluted bulk-seq")
# plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
# plt.colorbar(im)
# ax.set_title("Expression correlation")
# fig.tight_layout()
# plt.savefig("Try2.png")
# generate_sc_meta, generate_sc_data = model.load_vae_and_generate(
#     input_bulk_path='tutorial/data/example_data/demo1/demo1_bulk.csv',
#     input_sc_data_path='tutorial/data/example_data/demo1/demo1_sc_data.csv',
#     input_sc_meta_path='tutorial/data/example_data/demo1/demo1_sc_meta.csv',
#     input_st_data_path='tutorial/data/example_data/demo1/demo1_st_data.csv',
#     input_st_meta_path='tutorial/data/example_data/demo1/demo1_st_meta.csv',
#     vae_load_dir='tutorial/data/example_data/demo1/predata/save_model/demo1_vae.pth',
#     generate_save_dir='tutorial/data/example_data/demo1/predata/output',
#     generate_save_name='demo1_new',
#     ratio_num=1,
#     top_marker_num=500)





# df_meta, df_data = model.train_df_and_spatial_deconvolution(
#     generate_sc_meta,
#     generate_sc_data,
#     input_st_data_path='tutorial/data/example_data/demo1/demo1_st_data.csv',
#     input_st_meta_path='tutorial/data/example_data/demo1/demo1_st_meta.csv',
#     spot_num=500,
#     cell_num=10,
#     df_save_dir='tutorial/data/example_data/demo1/predata/save_model/',
#     df_save_name='deom1_df',
#     map_save_dir='tutorial/data/example_data/demo1/result',
#     map_save_name='demo1',
#     top_marker_num=500,
#     marker_used=True,
#     k=10)
#
# # Spatial mapping with single cell resolution
# ct_type = list(df_meta['Cell_type'].unique())
# color = ["#EF4444", "#10B981", '#14B8A6', "#7C3AED", '#3B82F6', '#64748B', "#D946EF",
#          '#F59E0B', '#F97316', '#7DD3FC', "#F472B6", "#EC4899"]
#
# fig, ax = plt.subplots(figsize=(8,8))
# for i in range(len(ct_type)):
#     ax.scatter(df_meta.loc[df_meta.Cell_type == ct_type[i], 'Cell_xcoord'],
#                df_meta.loc[df_meta.Cell_type == ct_type[i], 'Cell_ycoord'],
#                color = color[i], label = ct_type[i], s = 15)
#
#
# plt.title("Spatial mapping with single cell resolution")
# plt.xlabel("Cell_xcoord")
# plt.ylabel("Cell_ycoord")
# plt.legend(bbox_to_anchor=(1, 0.2), loc=3, borderaxespad=0, frameon=False, fontsize=15)
# plt.savefig("Try2.png")
