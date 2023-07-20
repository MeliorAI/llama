from typing import Any
from typing import Dict
from typing import List

import chromadb
import fire

from chromadb.types import Collection
from chromadb.utils import embedding_functions
from collections import defaultdict
from funcy import chunks
from hashlib import sha1
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import PDFMinerLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich import print as rprint
from tqdm.auto import tqdm


from rich.console import Console

console = Console()


DEFAULT_COLLECTION_NAME = 'llama-qa'
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 500


def file_sha1(filename):
    h = sha1()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])

    return str(h.hexdigest())


class NoIndexError(Exception):
    pass


class RetrieverNotInitialized(Exception):
    pass


class FilteredRetrievalQA(RetrievalQA):
    """🏗️ FilteredRetrievalQA overrides the `RetrievalQA` class to expose the
    filtering options of the underlying vector store (i.e.: ChromaDB)
    """

    def _get_docs(self, question: str, filters: Dict[str, Any]) -> List[Document]:
        # Ideally we would override also the retriever so we retain all the different
        # search types. Instead of:
        # 'return self.retriever.get_relevant_documents(question)""
        # For now:
        rprint(f"🤏🏼 Filtering by: {filters}")
        return self.retriever.vectorstore.similarity_search(question, filter=filters)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: CallbackManagerForChainRun | None = None,
    ) -> Dict[str, Any]:
        """Run get_relevant_text and llm on input query.

        If chain has 'return_source_documents' as 'True', returns
        the retrieved documents as well under the key 'source_documents'.

        Example:
        .. code-block:: python

        res = indexqa({'query': 'This is my query'})
        answer, docs = res['result'], res['source_documents']
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        docs = self._get_docs(question, filters=inputs["filters"])

        # rprint("=================================")
        # rprint(docs[0].metadata['type'])
        # rprint(f"pages: {[d.metadata['page'] for d in docs]}")
        # rprint("=================================")

        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


class IndexClient:
    def __init__(
        self,
        embedding_model_id:str = 'all-MiniLM-L6-v2',
        chroma_uri: str = 'localhost:8000',
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
                self.collection_name,
                embedding_function=self.embeddings
            )
            rprint(f"🥳 Collection '{self.collection_name}' found!")
            rprint(f"Embedding function: {col._embedding_function}")
            rprint(f"{col} | entries={col.count()}")
        except Exception as e:
            rprint(f"⚠️ [yellow]Collection not found. {e}[/yellow]")
            col = self.db_client.create_collection(
                self.collection_name,
                embedding_function=self.embeddings
            )

        return col

    def _insert_doc_in_db(
        self, doc_chunks: List[Document], doc_id:str, batch_size: int = 2
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
        return set(list(map(lambda m: m["id"], col.get()["metadatas"])))

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
            chunk.metadata['chunk'] = chunk_counter[sig]
            chunk_counter[sig] += 1

        rprint(
            f"🔪 Split into {len(doc_chunks)} chunks of text "
            f"(max. {self.chunk_size} tokens each)"
        )

        return self._insert_doc_in_db(doc_chunks, doc_id=doc_meta["id"])


def index(file_path:str):
    if not file_path.endswith(".pdf"):
        rprint("💥 [red]Only PDFs supported for now[/red]")
        exit(1)

    sha = file_sha1(file_path)
    pages = PDFMinerLoader(file_path).load_and_split()

    indexer = IndexClient()
    indexer.index_document(pages, doc_meta={"id": sha})


if __name__ == "__main__":
    fire.Fire(index)
