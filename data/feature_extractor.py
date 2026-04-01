import torch
import numpy as np
import pandas as pd
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_FILE = '/content/drive/MyDrive/BrainGen-Agent/data/borzoi_ready_sequences.pkl'
OUTPUT_FILE = '/content/drive/MyDrive/BrainGen-Agent/data/genomic_embeddings_eg.pt'

def check_gpu():
    logger.info("检查GPU可用性...")
    
    if not torch.cuda.is_available():
        raise RuntimeError("GPU不可用！Borzoi模型的100kb推断在CPU上极慢，请启用GPU运行时。")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    logger.info(f"检测到GPU: {gpu_name}")
    logger.info(f"GPU显存: {gpu_memory:.2f} GB")
    
    torch.cuda.empty_cache()

def one_hot_encode(sequence: str) -> np.ndarray:
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
    
    return onehot

def load_borzoi_model():
    logger.info("加载Borzoi模型...")
    
    from borzoi_pytorch import Borzoi
    
    model = Borzoi.from_pretrained('johahi/borzoi-replicate-0')
    model = model.cuda()
    model.eval()
    
    logger.info("Borzoi模型加载成功")
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def extract_embedding(model, sequence: str) -> torch.Tensor:
    onehot = one_hot_encode(sequence)
    onehot_tensor = torch.tensor(onehot, dtype=torch.float32).unsqueeze(0).cuda()
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            onehot_tensor = onehot_tensor.permute(0, 2, 1)
            outputs = model(onehot_tensor, return_embeddings=True)
            
            if isinstance(outputs, dict):
                embeddings = outputs.get('embeddings')
            elif isinstance(outputs, tuple):
                embeddings = outputs[0]
            else:
                embeddings = outputs
            
            if len(embeddings.shape) == 3:
                embedding = embeddings.mean(dim=1)
            elif len(embeddings.shape) == 4:
                embedding = embeddings.mean(dim=[1, 2])
            else:
                embedding = embeddings
            
            embedding = embedding.squeeze()
    
    return embedding.cpu()

def process_sequences():
    check_gpu()
    
    model = load_borzoi_model()
    
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
            embedding = extract_embedding(model, sequence)
            
            embeddings_dict[rsid] = embedding
            
            logger.info(f"  特征维度: {embedding.shape}")
            
        except Exception as e:
            logger.error(f"处理 {rsid} 失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
