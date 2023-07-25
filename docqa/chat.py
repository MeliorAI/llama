import chromadb
import fire
import torch

from chromadb.utils import embedding_functions
from llama import Llama
from rich import print as rprint

from docqa.constants import DEFAULT_EMBEDDING_MODEL
from docqa.constants import DEFAULT_COLLECTION_NAME
from docqa.constants import DEFAULT_CHROMA_URI
from docqa.constants import DEFAULT_CHUNK_SIZE
from docqa.constants import DEFAULT_CHUNK_OVERLAP
from docqa.constants import DEFAULT_TOKENIZER_PATH
from docqa.constants import DEFAULT_TEMPERATURE
from docqa.constants import DEFAULT_CKPT_DIR
from docqa.constants import DEFAULT_TOP_P
from docqa.constants import DEFAULT_MAX_SEQ_LEN
from docqa.constants import DEFAULT_MAX_BATCH_SIZE
from docqa.constants import DEFAULT_MAX_GEN_LEN
from docqa.constants import DEFAULT_N_RESULTS


def run(
    embedding_model_id:str = DEFAULT_EMBEDDING_MODEL,
    ckpt_dir: str = DEFAULT_CKPT_DIR,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    max_gen_len: int = DEFAULT_MAX_GEN_LEN,
    chroma_uri:str = DEFAULT_CHROMA_URI,
    chroma_collection:str = DEFAULT_COLLECTION_NAME,
    n_results:int = DEFAULT_N_RESULTS,
):
    """Run a Document QA 🦙 chat"""

    rprint(
        f"ℹ️ Cuda support: {torch.cuda.is_available()} "
        f"({torch.cuda.device_count()} devices)"
    )

    rprint("🧬 Initializing DB")
    try:
        emb_f = embedding_functions.SentenceTransformerEmbeddingFunction(embedding_model_id)

        host, port = chroma_uri.split(":")
        client = chromadb.HttpClient(host=host, port=int(port))
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
        # context = ""
        # sources = []

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

        for _, result in zip(dialogs, results):
            rprint(f"🤖: {result['generation']['content']}")
            rprint("[dim]------[/dim]")
            for src_meta, src_content in zip(sources, hits['documents'][0]):
                rprint(f"[dim]Sources:{src_meta}[/dim]")
                rprint(f"[dim]Sources:{src_content[:50]}[/dim]")

        rprint("\n[magenta]==================================[/magenta]\n")


if __name__ == "__main__":
    fire.Fire(run)
