import pickle
import requests
import time
import logging
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_FILE = '/content/drive/MyDrive/BrainGen-Agent/data/significant_hippocampus_snps.pkl'
OUTPUT_FILE = '/content/drive/MyDrive/BrainGen-Agent/data/borzoi_ready_sequences.pkl'

VARIATION_API_BASE = 'https://rest.ensembl.org/variation/human'
SEQUENCE_API_BASE = 'https://rest.ensembl.org/sequence/region/human'

FLANK_SIZE = 50000
MAX_RETRIES = 3
RETRY_DELAY = 2

def load_snps() -> Dict:
    logger.info(f"加载SNP数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        import pandas as pd
        df = pd.DataFrame(list(data.items()), columns=['RSID', 'Effect_Beta'])
    else:
        df = data
    
    logger.info(f"共加载 {len(df)} 个SNP")
    return df

def get_snp_coordinates(rsid: str) -> Optional[Dict]:
    url = f"{VARIATION_API_BASE}/{rsid}?content-type=application/json"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            
            data = response.json()
            
            if 'mappings' in data and len(data['mappings']) > 0:
                mapping = data['mappings'][0]
                chrom = mapping.get('seq_region_name')
                position = mapping.get('start')
                
                if chrom and position:
                    return {
                        'chromosome': chrom,
                        'position': position
                    }
            
            logger.warning(f"RSID {rsid} 未找到有效的坐标信息")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"获取 {rsid} 坐标失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"获取 {rsid} 坐标最终失败")
                return None

def get_sequence(chrom: str, position: int) -> Optional[str]:
    start = position - FLANK_SIZE
    end = position + FLANK_SIZE
    
    url = f"{SEQUENCE_API_BASE}/{chrom}:{start}..{end}?coord_system_version=GRCh38"
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers={"Content-Type": "text/plain"})
            response.raise_for_status()
            
            sequence = response.text.strip()
            
            if sequence and len(sequence) > 0:
                return sequence
            
            logger.warning(f"染色体 {chrom} 位置 {position} 未获取到序列")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"获取序列失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"获取序列最终失败: {chrom}:{position}")
                return None

def build_sequences():
    df = load_snps()
    
    results = []
    total = len(df)
    
    logger.info(f"开始处理 {total} 个SNP...")
    
    for idx, row in df.iterrows():
        rsid = row['RSID']
        effect_beta = row['Effect_Beta']
        
        logger.info(f"正在抓取 {rsid} ({idx + 1}/{total})...")
        
        coords = get_snp_coordinates(rsid)
        
        if coords is None:
            logger.warning(f"跳过 {rsid}: 无法获取坐标")
            time.sleep(1)
            continue
        
        chrom = coords['chromosome']
        position = coords['position']
        
        logger.info(f"  坐标: chr{chrom}:{position}")
        
        sequence = get_sequence(chrom, position)
        
        if sequence is None:
            logger.warning(f"跳过 {rsid}: 无法获取序列")
            time.sleep(1)
            continue
        
        logger.info(f"  序列长度: {len(sequence)} bp")
        
        results.append({
            'RSID': rsid,
            'Effect_Beta': effect_beta,
            'Chromosome': chrom,
            'Position': position,
            'Sequence_100kb': sequence
        })
        
        time.sleep(1)
    
    logger.info(f"成功处理 {len(results)}/{total} 个SNP")
    
    import pandas as pd
    result_df = pd.DataFrame(results)
    
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(result_df, f)
    
    logger.info(f"结果已保存到: {OUTPUT_FILE}")
    
    return result_df

if __name__ == "__main__":
    build_sequences()
