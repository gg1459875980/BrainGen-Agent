import pandas as pd
import logging
import gzip
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INPUT_FILE = 'raw/ENIGMA2_MeanHippocampus_Combined_GenomeControlled_Jan23.tbl.gz'
OUTPUT_FILE = 'significant_hippocampus_snps.pkl'
CHUNKSIZE = 100000
PVAL_THRESHOLD = 5e-8

COL_MARKER = 'RSID'
COL_EFFECT = 'Effect_Beta'
COL_PVAL = 'Pvalue'

def parse_enigma_data():
    significant_snps = []
    chunk_count = 0
    
    logger.info(f"开始处理文件: {INPUT_FILE}")
    logger.info(f"使用块大小: {CHUNKSIZE}")
    logger.info(f"P值阈值: {PVAL_THRESHOLD}")
    
    try:
        for chunk in pd.read_table(INPUT_FILE, compression='gzip', chunksize=CHUNKSIZE, sep=r'\s+'):
            chunk_count += 1
            logger.info(f"正在处理第 {chunk_count} 块数据...")
            
            if COL_MARKER not in chunk.columns or COL_EFFECT not in chunk.columns or COL_PVAL not in chunk.columns:
                logger.error(f"缺少必要列！当前块列名: {chunk.columns.tolist()}")
                continue
            
            filtered = chunk[chunk[COL_PVAL] < PVAL_THRESHOLD]
            
            if not filtered.empty:
                significant_snps.append(filtered[[COL_MARKER, COL_EFFECT]])
                logger.info(f"第 {chunk_count} 块找到 {len(filtered)} 个显著 SNP")
        
        if significant_snps:
            result_df = pd.concat(significant_snps, ignore_index=True)
            logger.info(f"共找到 {len(result_df)} 个显著 SNP")
            
            result_dict = dict(zip(result_df[COL_MARKER], result_df[COL_EFFECT]))
            
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(result_dict, f)
            
            logger.info(f"结果已保存到: {OUTPUT_FILE}")
            return result_dict
        else:
            logger.info("未找到显著 SNP")
            return {}
            
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    parse_enigma_data()
