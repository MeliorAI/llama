import chromadb
import fire

from chromadb.utils import embedding_functions
from llama import Llama
from rich import print as rprint
from typing import Optional


def main(
    embedding_model_id:str = 'all-MiniLM-L6-v2',
    ckpt_dir: str = 'llama-2-7b-chat',
    tokenizer_path: str = 'tokenizer.model',
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    chroma_host:str = 'localhost',
    chroma_port:str = '8000',
    chroma_collection:str = 'llama-qa'
):
    rprint("🧬 Initializing DB")
    emb_f = embedding_functions.SentenceTransformerEmbeddingFunction(embedding_model_id)
    client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
    col = client.get_collection(chroma_collection, embedding_function=emb_f)

    rprint("🦙 Initializing Llama")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    while query := input("🗣️: "):

        hits = col.query(query_texts=[query])
        sources = hits['metadata']
        evidences = hits['documents']

        dialog = [
                {"role": "system", "content": "You are given the following pieces of context to use to answer"},
                *[{"role": "system", "content": evidence} for evidence in evidences],
                {"role": "user", "content": query},
        ],

        results = generator.chat_completion(
            [dialog],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            print(f"🤖: {result['generation']['content']}")
            print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
