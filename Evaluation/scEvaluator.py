import numpy as np
import scanpy as sc
import anndata as ad
from scipy import stats, sparse, io
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch.autograd import Variable
import scib
from tqdm import tqdm

class SingleCellEvaluator:
    def __init__(self, real_data, gen_data):
        """
        Parameters:
        -----------
        real_data : np.ndarray or anndata.AnnData
            真实数据(可以是numpy数组或AnnData对象)
        gen_data : np.ndarray or anndata.AnnData  
            生成数据(可以是numpy数组或AnnData对象)
        """
        if isinstance(real_data, ad.AnnData):
            self.real_data = real_data.X.toarray() if sparse.issparse(real_data.X) else real_data.X
            self.real_adata = real_data
        else:
            self.real_data = real_data
            self.real_adata = ad.AnnData(real_data)
            
        if isinstance(gen_data, ad.AnnData):
            self.gen_data = gen_data.X.toarray() if sparse.issparse(gen_data.X) else gen_data.X
            self.gen_adata = gen_data
        else:
            self.gen_data = gen_data
            self.gen_adata = ad.AnnData(gen_data)
        
        # Filtering values less than 0.2
        self.gen_data[self.gen_data < 0.2] = 0  
        
        if self.real_data.shape[1] != self.gen_data.shape[1]:
            raise ValueError(f"Gene numbers don't match: {self.real_data.shape[1]} vs {self.gen_data.shape[1]}")


    def preprocess_if_needed(self, adata):
        sc.pp.filter_cells(adata, min_genes=10)
        sc.pp.filter_genes(adata, min_cells=3)
        if 'n_counts' not in adata.obs:
            sc.pp.normalize_total(adata, target_sum=1e4)
        if 'log1p' not in adata.uns:
            sc.pp.log1p(adata)
        return adata

    def calculate_correlations(self):
        spearman = stats.spearmanr(self.real_data.mean(axis=0), 
                                  self.gen_data.mean(axis=0)).correlation
        pearson = np.corrcoef(self.real_data.mean(axis=0), 
                             self.gen_data.mean(axis=0))[0][1]
        return {
            'spearman': spearman, 
            'pearson': pearson
        }

    def calculate_mmd(self, batch_size=100, sample_size=1000, device='cpu'):
        """
        Parameters:
        -----------
        batch_size : int
        sample_size : int
        device : str
            'cpu' 或 'cuda' cuda还没试过
        Returns:
        --------
        float : MMD Score
        """
        def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            n_samples = int(source.size()[0])+int(target.size()[0])
            total = torch.cat([source, target], dim=0)

            total0 = total.unsqueeze(0).expand(int(total.size(0)), 
                                            int(total.size(0)), 
                                            int(total.size(1)))
            total1 = total.unsqueeze(1).expand(int(total.size(0)), 
                                            int(total.size(0)), 
                                            int(total.size(1)))
            
            L2_distance = torch.zeros((total.size(0), total.size(0))).to(device)
            
            for i in tqdm(range(0, total.size(0), batch_size), desc="Computing MMD"):
                end_i = min(i + batch_size, total.size(0))
                for j in range(0, total.size(0), batch_size):
                    end_j = min(j + batch_size, total.size(0))
                    
                    batch_diff = (total0[i:end_i, j:end_j] - total1[i:end_i, j:end_j])
                    L2_distance[i:end_i, j:end_j] = (batch_diff ** 2).sum(2)

            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            
            bandwidth /= kernel_mul ** (kernel_num // 2)
            bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
            
            kernel_val = torch.zeros_like(L2_distance)
            for bandwidth_temp in bandwidth_list:
                kernel_val += torch.exp(-L2_distance / bandwidth_temp)
            
            return kernel_val

        def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            batch_size = int(source.size()[0])
            kernels = gaussian_kernel(source, target, 
                                    kernel_mul=kernel_mul,
                                    kernel_num=kernel_num,
                                    fix_sigma=fix_sigma)
            
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            
            loss = torch.mean(XX + YY - XY - YX)
            return loss

        idx_real = np.random.choice(self.real_data.shape[0], min(sample_size, self.real_data.shape[0]), replace=False)
        idx_gen = np.random.choice(self.gen_data.shape[0], min(sample_size, self.gen_data.shape[0]), replace=False)
        
        X = torch.Tensor(self.real_data[idx_real]).to(device)
        Y = torch.Tensor(self.gen_data[idx_gen]).to(device)
        
        X, Y = Variable(X), Variable(Y)
        
        try:
            mmd_score = mmd_rbf(X, Y)
            return mmd_score.item()
        except RuntimeError as e:
            print(f"Warning: MMD Failed: {str(e)}")
            print("Try reducing sample_size or batch_size")
            return None

    def calculate_ilisi(self, n_neighbors=10, n_pcs=20):
        adata_combined = ad.concat(
            [self.real_adata, self.gen_adata],
            join='inner',
            label="batch",
            keys=['real', 'gen']
        )
        
        sc.pp.neighbors(adata_combined, n_neighbors=n_neighbors, n_pcs=n_pcs)
        ilisi_score = scib.me.ilisi_graph(adata_combined, batch_key="batch", type_="knn")
        
        return ilisi_score

    def calculate_knn_metrics(self, n_neighbors=5, test_size=0.3, random_state=1):
        data = np.concatenate((self.real_data, self.gen_data), axis=0)
        labels = np.concatenate((np.ones(len(self.real_data)), 
                              np.zeros(len(self.gen_data))))

        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, 
            test_size=test_size, 
            random_state=random_state
        )

        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        classifier.fit(X_train, y_train)
        
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)
        
        return {
            'accuracy': accuracy,
            'auc': auc
        }
    
    def evaluate_all(self):
        results = {}
        results['correlations'] = self.calculate_correlations()
        results['mmd'] = self.calculate_mmd()
        # results['ilisi'] = self.calculate_ilisi() # 有bug，解决起来比较麻烦，需要现场编译来安装这个库，可以不测
        results['knn_metrics'] = self.calculate_knn_metrics()
        
        return results


# 使用示例
if __name__ == "__main__":
    print("Loading data")
    real_data = io.mmread(f"/root/autodl-tmp/b2sc/saved_files/nonepi/final_batches/batch_4_original.mtx").toarray()
    gen_data = io.mmread(f"/root/autodl-tmp/b2sc/saved_files/nonepi/final_batches/batch_4_reconstructed.mtx").toarray()
    # real_adata = sc.read_h5ad('path/to/real_data.h5ad')
    # gen_adata = sc.read_h5ad('path/to/gen_data.h5ad')
    
    print("Creating evaluator...")
    evaluator = SingleCellEvaluator(real_data, gen_data)
    
    print("Running evaluation...")
    try:
        results = evaluator.evaluate_all()
        
        print("\nEvaluation Results:")
        for metric, value in results.items():
            if value is not None:  
                # 只打印成功的评估结果
                print(f"{metric}:")
                if isinstance(value, dict):
                    for k, v in value.items():
                        print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {value:.4f}")
            else:
                print(f"{metric}: Failed")
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")