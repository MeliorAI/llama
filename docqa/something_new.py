import glob
import os
from collections import defaultdict
from typing import Any, Dict, List

import chromadb
import fire
from chromadb.types import Collection
from chromadb.utils import embedding_functions
from docqa import file_sha1
from docqa.constants import (DEFAULT_CHROMA_URI, DEFAULT_CHUNK_OVERLAP,
                             DEFAULT_CHUNK_SIZE, DEFAULT_COLLECTION_NAME,
                             DEFAULT_EMBEDDING_MODEL, DEFAULT_N_RESULTS)
from funcy import chunks
from langchain.docstore.document import Document
from langchain.document_loaders import PDFMinerLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich import print as rprint
from rich.console import Console
from tabulate import tabulate
from tqdm.auto import tqdm

console = Console()


class IndexClient:
    def __init__(
        self,
        embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
        chroma_uri: str = DEFAULT_CHROMA_URI,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        # ✂️ Text chunking configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 🎛️ Retrieval configuration (hard-coded for now)
        self.target_source_chunks = 4
        self.chain_type = "stuff"

        # Embedding models
        self.embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_id
        )

        # 🌈 ChromaDB
        self._init_db(chroma_uri, collection_name)

    def _init_db(self, chroma_uri, collection_name):
        rprint("🔧 Initializing vectorstore")
        self.chroma_uri = chroma_uri
        self.collection_name = collection_name

        host, port = chroma_uri.split(":")

        self.db_client = chromadb.HttpClient(host=host, port=int(port))
        self.db_col = self._init_collection()

    def _init_collection(self) -> Collection:
        try:
            col = self.db_client.get_collection(
                self.collection_name, embedding_function=self.embeddings
            )
            rprint(f"🥳 Collection '{self.collection_name}' found!")
            rprint(f"Embedding function: {col._embedding_function}")
            rprint(f"{col} | entries={col.count()}")
        except Exception as e:
            rprint(f"⚠️ [yellow]Collection not found. {e}[/yellow]")
            col = self.db_client.create_collection(
                self.collection_name, embedding_function=self.embeddings
            )

        return col

    def _insert_doc_in_db(
        self, doc_chunks: List[Document], doc_id: str, batch_size: int = 2
    ) -> List[str]:
        inserted = []
        try:
            # NOTE: Insert in batches, otherwise we can get a disconnect from the server
            n_batches = len(doc_chunks) // batch_size
            for batch in tqdm(chunks(batch_size, doc_chunks), total=n_batches):
                self.db_col.add(
                    documents=[d.page_content for d in batch],
                    metadatas=[d.metadata for d in batch],
                    ids=[
                        f"{doc_id}/{d.metadata['page']}/{d.metadata['chunk']}"
                        for d in batch
                    ],
                )
        except Exception as e:
            rprint(f"💥 [red]Error inserting document in DB: {e}[/red]")
        else:
            rprint(f"✅ [green]Document inserted![/green]")

    def list_documents(self):
        col = self.db_client.get_collection(self.collection_name)
        return set(list(map(lambda m: m.get("id"), col.get()["metadatas"])))

    def delete_documents(self, doc_ids: List[str] = [], filters: Dict[str, Any] = {}):
        col = self.db_client.get_collection(self.collection_name)
        col.delete(ids=doc_ids, where=filters)

    def index_document(
        self, doc_pages: List[Document], doc_meta: Dict[str, Any] = {}
    ) -> List[str]:
        rprint(f"📚️ Adding {len(doc_pages)} document pages")
        documents = [
            Document(page_content=d.page_content, metadata={"page": i, **d.metadata})
            for i, d in enumerate(doc_pages)
        ]

        # Chunk pages into smaller pieces
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        doc_chunks = text_splitter.split_documents(documents)
        chunk_counter = defaultdict(int)
        for chunk in doc_chunks:
            sig = str(chunk.metadata)
            chunk.metadata["chunk"] = chunk_counter[sig]
            chunk_counter[sig] += 1

        rprint(
            f"🔪 Split into {len(doc_chunks)} chunks of text "
            f"(max. {self.chunk_size} tokens each)"
        )

        return self._insert_doc_in_db(doc_chunks, doc_id=doc_meta["id"])


def pdf(
    file_path: str,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
    chroma_uri: str = DEFAULT_CHROMA_URI,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Index a PDF document given its file path"""
    if not file_path.endswith(".pdf"):
        rprint("💥 [red]Only PDFs supported for now[/red]")
        exit(1)

    sha = file_sha1(file_path)
    pages = PDFMinerLoader(file_path).load_and_split()

    indexer = IndexClient(embedding_model_id, chroma_uri, collection_name)
    indexer.index_document(pages, doc_meta={"id": sha})


def dir(
    data_dir: str,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
    chroma_uri: str = DEFAULT_CHROMA_URI,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Index all PDFs document ina  given directory path"""
    indexer = IndexClient(embedding_model_id, chroma_uri, collection_name)

    for pdf_path in glob.glob(os.path.join(data_dir, "*.pdf")):
        rprint(f"⏳️ Indexing {pdf_path}")
        sha = file_sha1(pdf_path)
        pages = PDFMinerLoader(pdf_path).load_and_split()
        indexer.index_document(pages, doc_meta={"id": sha})


def list_collections(
    chroma_uri: str = DEFAULT_CHROMA_URI,
):
    """Prints a table of chromaDB collections and a list of its documents"""
    host, port = chroma_uri.split(":")
    client = chromadb.HttpClient(host=host, port=int(port))

    rprint("\n[magenta][bold] --- COLLECTIONS --- [/bold][/magenta]")
    cols = []
    if collections := client.list_collections():
        for col in collections:
            cols.append((col.name, col.id, col.metadata, col.count()))

        print(
            tabulate(
                cols, headers=["Name", "ID", "metadata", "entries"], tablefmt="grid"
            )
        )

    else:
        rprint(f"👀 No collections found!")


def list_docs(
    collection_name: str,
    chroma_uri: str = DEFAULT_CHROMA_URI,
):
    """Print a table with all documents in the given collection"""
    host, port = chroma_uri.split(":")
    client = chromadb.HttpClient(host=host, port=int(port))

    rprint("\n[blue][bold] --- DOCUMENTS --- [/bold][/blue]")
    col = client.get_collection(collection_name)
    if metas := col.get()["metadatas"]:
        sources = set([m.get("source") or m.get("id") for m in metas])
        docs = [(src,) for src in sources]

        print(tabulate(docs, headers=[f"{col.name} docs"], tablefmt="grid"))
    else:
        rprint(f"👀 No documents found!")


def clear(
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
    chroma_uri: str = DEFAULT_CHROMA_URI,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Clear a ChromaDB collection given its name and embedding function"""
    try:
        indexer = IndexClient(embedding_model_id, chroma_uri, collection_name)
        rprint(f"BEFORE: {indexer.list_documents()}")
        indexer.delete_documents()
        rprint(f"AFTER: {indexer.list_documents()}")
    except Exception as e:
        rprint(f"[red]{e}[/red]")
    else:
        rprint("✅ Done!")


def delete_collection(
    chroma_uri: str = DEFAULT_CHROMA_URI,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Delete a given chromaDB collection given its name"""
    try:
        host, port = chroma_uri.split(":")
        client = chromadb.HttpClient(host=host, port=int(port))
        client.delete_collection(collection_name)
    except Exception as e:
        rprint(f"[red]{e}[/red]")
    else:
        rprint("✅ Done!")


def search(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL,
    chroma_uri: str = DEFAULT_CHROMA_URI,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """Search a given ChromaDB collection against a given textual query"""
    # Chroma DB
    chroma_host, chroma_port = chroma_uri.split(":")
    client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))

    # Chroma collection
    emb_f = embedding_functions.SentenceTransformerEmbeddingFunction(embedding_model_id)
    col = client.get_collection(collection_name, embedding_function=emb_f)

    # Search
    res = col.query(query_texts=[query], n_results=n_results)

    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        content = doc.replace("\n", "")
        rprint(f"ℹ️ [dim]{meta}[/dim]")
        rprint(f"📃 {content}")
        print("--------------\n")


if __name__ == "__main__":
    fire.Fire(
        {
            "pdf": pdf,
            "dir": dir,
            "clear": clear,
            "search": search,
            "lsc": list_collections,
            "lsd": list_docs,
            "delete-collection": delete_collection,
        }
    )
