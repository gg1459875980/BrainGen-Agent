import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import logging
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SNP_FILE = '/content/drive/MyDrive/BrainGen-Agent/data/significant_hippocampus_snps.pkl'
GENOMIC_FILE = '/content/drive/MyDrive/BrainGen-Agent/data/genomic_embeddings_eg.pt'
OUTPUT_FILE = '/content/drive/MyDrive/BrainGen-Agent/data/alignment_heads.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_align_data() -> Tuple[torch.Tensor, torch.Tensor]:
    logger.info("加载数据...")
    
    with open(SNP_FILE, 'rb') as f:
        snp_data = pickle.load(f)
    
    if isinstance(snp_data, dict):
        import pandas as pd
        snp_df = pd.DataFrame(list(snp_data.items()), columns=['RSID', 'Effect_Beta'])
    else:
        snp_df = snp_data
    
    genomic_data = torch.load(GENOMIC_FILE, map_location='cpu')
    
    logger.info(f"SNP数据 RSID数量: {len(snp_df)}")
    logger.info(f"基因组数据 RSID数量: {len(genomic_data)}")
    
    common_rsids = set(snp_df['RSID']) & set(genomic_data.keys())
    logger.info(f"交集 RSID数量: {len(common_rsids)}")
    
    gene_features = []
    pheno_features = []
    
    for rsid in common_rsids:
        gene_feature = genomic_data[rsid]
        pheno_value = snp_df[snp_df['RSID'] == rsid]['Effect_Beta'].values[0]
        
        gene_features.append(gene_feature)
        pheno_features.append(pheno_value)
    
    gene_tensor = torch.stack(gene_features)
    pheno_tensor = torch.tensor(pheno_features, dtype=torch.float32).unsqueeze(1)
    
    logger.info(f"基因特征形状: {gene_tensor.shape}")
    logger.info(f"表型特征形状: {pheno_tensor.shape}")
    
    return gene_tensor, pheno_tensor

class GeneProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

class PhenotypeProjector(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 16, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

def info_nce_loss(z_gene: torch.Tensor, z_pheno: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    batch_size = z_gene.shape[0]
    
    sim_matrix = torch.matmul(z_gene, z_pheno.T) / temperature
    
    labels = torch.arange(batch_size, device=z_gene.device)
    
    loss_g2p = F.cross_entropy(sim_matrix, labels)
    loss_p2g = F.cross_entropy(sim_matrix.T, labels)
    
    loss = (loss_g2p + loss_p2g) / 2
    
    return loss

def train():
    gene_tensor, pheno_tensor = load_and_align_data()
    gene_tensor = gene_tensor.to(DEVICE)
    pheno_tensor = pheno_tensor.to(DEVICE)
    
    input_dim = gene_tensor.shape[1]
    logger.info(f"基因特征输入维度: {input_dim}")
    
    gene_projector = GeneProjector(input_dim=input_dim).to(DEVICE)
    pheno_projector = PhenotypeProjector().to(DEVICE)
    
    optimizer = torch.optim.Adam(
        list(gene_projector.parameters()) + list(pheno_projector.parameters()),
        lr=1e-3
    )
    
    num_epochs = 100
    
    logger.info("开始训练...")
    
    for epoch in range(num_epochs):
        gene_projector.train()
        pheno_projector.train()
        
        z_gene = gene_projector(gene_tensor)
        z_pheno = pheno_projector(pheno_tensor)
        
        loss = info_nce_loss(z_gene, z_pheno)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    logger.info("训练完成！")
    
    checkpoint = {
        'gene_projector': gene_projector.state_dict(),
        'pheno_projector': pheno_projector.state_dict()
    }
    
    torch.save(checkpoint, OUTPUT_FILE)
    logger.info(f"模型权重已保存到: {OUTPUT_FILE}")
    
    return gene_projector, pheno_projector

if __name__ == "__main__":
    train()
