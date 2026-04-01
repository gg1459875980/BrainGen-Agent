import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_FILE = '/content/drive/MyDrive/BrainGen-Agent/data/borzoi_ready_sequences.pkl'
OUTPUT_FILE = '/content/drive/MyDrive/BrainGen-Agent/data/genomic_embeddings_eg.pt'

DEVICE = None
MODEL = None

def check_gpu():
    logger.info("检查GPU可用性...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("GPU不可用！Borzoi模型的100kb推断在CPU上极慢，请启用GPU运行时。")
    
    global DEVICE
    DEVICE = torch.device('cuda')
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    logger.info(f"检测到GPU: {gpu_name}")
    logger.info(f"GPU显存: {gpu_memory:.2f} GB")
    
    torch.cuda.empty_cache()

def dna_to_onehot(sequence: str) -> torch.Tensor:
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'a': [1, 0, 0, 0],
        'c': [0, 1, 0, 0],
        'g': [0, 0, 1, 0],
        't': [0, 0, 0, 1],
        'N': [0, 0, 0, 0],
        'n': [0, 0, 0, 0]
    }
    
    onehot = np.zeros((len(sequence), 4), dtype=np.float32)
    
    for i, base in enumerate(sequence):
        if base in mapping:
            onehot[i, :] = mapping[base]
    
    tensor = torch.tensor(onehot, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)
    
    return tensor

def load_borzoi_model():
    global MODEL
    
    logger.info("加载Borzoi模型...")
    
    try:
        from borzoi import Borzoi
        from borzoi import load_model
        
        MODEL = load_model()
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        
        logger.info("Borzoi模型加载成功")
        
        for param in MODEL.parameters():
            param.requires_grad = False
            
    except ImportError:
        logger.warning("未找到borzoi库，尝试使用enformer-pytorch...")
        
        try:
            from enformer_pytorch import Enformer
            from enformer_pytorch import load_enformer_weights
            
            MODEL = Enformer.from_pretrained('EleutherAI/enformer')
            MODEL = MODEL.to(DEVICE)
            MODEL.eval()
            
            logger.info("Enformer模型加载成功")
            
            for param in MODEL.parameters():
                param.requires_grad = False
                
        except ImportError:
            raise ImportError(
                "未找到borzoi或enformer-pytorch库。\n"
                "请安装: pip install borzoi 或 pip install enformer-pytorch"
            )

def extract_features(sequence: str) -> torch.Tensor:
    onehot = dna_to_onehot(sequence)
    onehot = onehot.to(DEVICE)
    
    with torch.no_grad():
        try:
            output = MODEL(onehot, return_embeddings=True)
            
            if isinstance(output, dict):
                embedding = output.get('embeddings', output.get('hidden_states'))
            elif isinstance(output, tuple):
                embedding = output[-1]
            else:
                embedding = output
            
            if len(embedding.shape) == 3:
                embedding = embedding.mean(dim=1)
            elif len(embedding.shape) == 4:
                embedding = embedding.mean(dim=[1, 2])
            
            embedding = embedding.squeeze()
            
        except Exception as e:
            logger.warning(f"标准提取失败，尝试备用方法: {str(e)}")
            
            with torch.no_grad():
                output = MODEL(onehot)
                
                if isinstance(output, tuple):
                    embedding = output[0]
                else:
                    embedding = output
                
                if len(embedding.shape) == 3:
                    embedding = embedding.mean(dim=1)
                elif len(embedding.shape) == 4:
                    embedding = embedding.mean(dim=[1, 2])
                
                embedding = embedding.squeeze()
    
    return embedding.cpu()

def process_sequences():
    check_gpu()
    
    load_borzoi_model()
    
    logger.info(f"加载序列数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"共加载 {len(data)} 个序列")
    
    embeddings_dict = {}
    total = len(data)
    
    logger.info("开始特征提取...")
    
    for idx, row in data.iterrows():
        rsid = row['RSID']
        sequence = row['Sequence_100kb']
        
        logger.info(f"处理 {rsid} ({idx + 1}/{total})...")
        logger.info(f"  序列长度: {len(sequence)} bp")
        
        try:
            embedding = extract_features(sequence)
            
            embeddings_dict[rsid] = embedding
            
            logger.info(f"  特征维度: {embedding.shape}")
            
        except Exception as e:
            logger.error(f"处理 {rsid} 失败: {str(e)}")
            continue
        
        torch.cuda.empty_cache()
        
        if (idx + 1) % 5 == 0:
            torch.cuda.empty_cache()
            logger.info(f"  显存已清理")
    
    logger.info(f"成功提取 {len(embeddings_dict)}/{total} 个特征")
    
    torch.save(embeddings_dict, OUTPUT_FILE)
    logger.info(f"结果已保存到: {OUTPUT_FILE}")
    
    return embeddings_dict

if __name__ == "__main__":
    process_sequences()
