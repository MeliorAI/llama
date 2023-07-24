# ChromaDB
DEFAULT_COLLECTION_NAME:str = 'llama-qa'
DEFAULT_CHROMA_URI:str = 'localhost:8000'
# text pre-processing
DEFAULT_CHUNK_SIZE:int = 1500
DEFAULT_CHUNK_OVERLAP:int = 500
DEFAULT_EMBEDDING_MODEL:str = 'all-mpnet-base-v2'
# Model
DEFAULT_CKPT_DIR: str = 'llama-2-7b-chat'
DEFAULT_TOKENIZER_PATH: str = 'tokenizer.model'
# Generation parameters
DEFAULT_TEMPERATURE: float = 0.6
DEFAULT_TOP_P: float = 0.9
DEFAULT_MAX_SEQ_LEN: int = 512
DEFAULT_MAX_BATCH_SIZE: int = 4
DEFAULT_MAX_GEN_LEN: int = None
DEFAULT_N_RESULTS:int = 3
