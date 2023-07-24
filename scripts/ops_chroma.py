import chromadb
import fire

from chromadb.utils import embedding_functions
from rich import print as rprint

def main(
    query:str,
    n_results:int = 4,
    embedding_model_id:str = 'all-MiniLM-L6-v2',
    chroma_host:str = 'localhost',
    chroma_port:str = '8000',
    chroma_collection:str = 'llama-qa'
):
    emb_f = embedding_functions.SentenceTransformerEmbeddingFunction()
    client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
    col = client.get_collection(chroma_collection, embedding_function=emb_f)
    res = col.query(query_texts=[query], n_results=n_results)

    rprint(res['metadatas'][0])


if __name__ == "__main__":
    fire.Fire(main)
