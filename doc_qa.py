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
    chroma_collection:str = 'llama-qa',
    n_results:int = 3,
):
    rprint("🧬 Initializing DB")
    try:
        emb_f = embedding_functions.SentenceTransformerEmbeddingFunction(embedding_model_id)
        client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
        col = client.get_collection(chroma_collection, embedding_function=emb_f)
    except Exception as e:
        rprint(f"💥 [red]Error initializing chroma db: {e}[/red]")
        exit(1)

    rprint("🦙 Initializing LlaMa")
    try:
        generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
    except Exception as e:
        rprint(f"💥 [red]Error initializing 🦙: {e}[/red]")
        exit(1)

    while query := input("🗣️: "):
        # Search
        hits = col.query(query_texts=[query], n_results=n_results)
        sources = hits['metadatas'][0]
        context =  "\nContext: ".join([
            "Excerpt {i}: {ctx}" for i, ctx in enumerate(hits['documents'][0])
        ])

        system_content = (
            "Use the following pieces of context to answer the question at the end. "
            "If you don't know the answer, just say that you don't know. "
            "Don't try to make up an answer."
            "When the question can be answered from the context also include which "
            "was the useful piece of information to answer."
        )

        # Compose the dialogue
        dialogs = [
            {
                "role": "system",
                "content": system_content
            },
            {"role": "user", "content": f"{context}\n{query}"},
        ]

        # Generate
        results = generator.chat_completion(
            [dialogs],  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        for dialog, result in zip(dialogs, results):
            rprint(f"🤖: {result['generation']['content']}")
            rprint("[dim]------[/dim]")
            # rprint(f"[dim]Sources:{sources}[/dim]")
            rprint(f"[dim]Sources:{hits['documents'][0]}[/dim]")
            rprint("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
