from typing import Any, Dict, List
import chromadb
import fire
import torch

from chromadb.utils import embedding_functions
from llama import Llama
from rich import print as rprint

from docqa.constants import DEFAULT_EMBEDDING_MODEL
from docqa.constants import DEFAULT_COLLECTION_NAME
from docqa.constants import DEFAULT_CHROMA_URI
from docqa.constants import DEFAULT_TOKENIZER_PATH
from docqa.constants import DEFAULT_TEMPERATURE
from docqa.constants import DEFAULT_CKPT_DIR
from docqa.constants import DEFAULT_TOP_P
from docqa.constants import DEFAULT_MAX_SEQ_LEN
from docqa.constants import DEFAULT_MAX_BATCH_SIZE
from docqa.constants import DEFAULT_MAX_GEN_LEN
from docqa.constants import DEFAULT_N_RESULTS


class Retriever:
    def __init__(
        self,
        embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
        chroma_uri: str = DEFAULT_CHROMA_URI,
        chroma_collection: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        rprint("🧬 Initializing DB")

        host, port = chroma_uri.split(":")
        self.emb_f = embedding_functions.SentenceTransformerEmbeddingFunction(
            embedding_model_id
        )
        self.client = chromadb.HttpClient(host=host, port=int(port))
        self.col = self.client.get_collection(
            chroma_collection, embedding_function=self.emb_f
        )

    def search(
        self,
        query: str,
        n_results: int = DEFAULT_N_RESULTS,
    ):
        # Search
        hits = self.col.query(query_texts=[query], n_results=n_results)
        metas = hits["metadatas"][0]
        sources = [ctx.replace("\n", "") for ctx in hits["documents"][0]]

        return metas, sources


class LlaMita:
    system_content = (
        "Use the following document excerpts to answer the question at the end. "
        "If you don't know the answer, just say that you don't know. "
        "When appropiate also include which was the useful piece of information"
    )

    def __init__(
        self,
        ckpt_dir: str = DEFAULT_CKPT_DIR,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    ) -> None:
        print("🦙 Initializing LlaMa")
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    def ask(
        self,
        sources: List[str],
        query: str,
        max_gen_len: int = DEFAULT_MAX_GEN_LEN,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> List[Dict[str, Any]]:
        # Compose the dialogue
        context = "\n".join([f"Excerpt {i+1}: {ctx}" for i, ctx in enumerate(sources)])
        dialogs = [
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": f"{context}\n{query}"},
        ]
        # Generate
        return self.generator.chat_completion(
            [dialogs],  # type: ignore
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )


def run(
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
    ckpt_dir: str = DEFAULT_CKPT_DIR,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    max_gen_len: int = DEFAULT_MAX_GEN_LEN,
    chroma_uri: str = DEFAULT_CHROMA_URI,
    chroma_collection: str = DEFAULT_COLLECTION_NAME,
    n_results: int = DEFAULT_N_RESULTS,
):
    """Run a Document QA 🦙 chat"""

    rprint(
        f"ℹ️  Cuda support: {torch.cuda.is_available()} "
        f"({torch.cuda.device_count()} devices)"
    )
    try:
        ret = Retriever(
            embedding_model_id,
            chroma_uri,
            chroma_collection,
        )
    except Exception as e:
        rprint(f"💥 [red]Error initializing chroma db: {e}[/red]")
        exit(1)

    try:
        llama = LlaMita(
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
        metas, sources = ret.search(query, n_results)
        # Generate
        results = llama.ask(
            sources,
            query,
            max_gen_len,
            temperature,
            top_p,
        )
        # Print
        for result in results:
            rprint(f"🦙:{result['generation']['content']}")
            rprint("ℹ️👇️ ----------- 👇️ℹ️")
            for i, (src_meta, src_content) in enumerate(zip(metas, sources)):
                rprint(f"[blue]---- Excerpt {i+1:02d} ----[/blue]")
                rprint(f"{src_meta}")
                rprint(f"📖:[dim]{src_content[:250]} [...][/dim]")

        rprint("\n[magenta]==================================[/magenta]\n")


if __name__ == "__main__":
    fire.Fire(run)
