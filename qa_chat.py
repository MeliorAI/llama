
class QAagent:

    def ask(
        self,
        query: str,
        filters: Dict[str, Any] = {},
        print_qa: bool = True,
        print_sources: bool = False,
    ):
        if self.db is None:
            raise NoIndexError(
                "No document vector database found. Has any document been indexed yet?"
            )
        if self.qa is None:
            raise RetrieverNotInitialized("QA Retriever has not been initialized.")

        with get_openai_callback() as cb:
            # Get the answer from the chain
            res = self.qa({"query": query, "filters": filters})
            answer, docs = res["result"], res["source_documents"]

            if print_qa:
                # Print the result
                rprint("\n\n> ❓️ [bold]Question:[/bold]")
                rprint(query)
                rprint("\n\n> 🤓 [bold]Answer:[/bold]")
                rprint(answer)

                # Print the relevant sources used for the answer
                if print_sources:
                    for i, document in enumerate(docs):
                        rprint(
                            f"\n> 📚️ [bold]'SOURCE {i}: {document.metadata['source']}':[/bold]"
                        )
                        rprint(f"[dim]{document.page_content}[/dim]\n")

            # print openAI consumption
            rprint(f"[dim]\n-----\n{cb}\n-----\n[dim]")

        return answer, docs
