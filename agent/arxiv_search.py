class ArxivSearch:
    def __init__(self):
        self.top_k_results = 3
        self.description = "查找论文"

    def reply(self, query):
        import arxiv
        try:
            results = arxiv.Search(query, max_results=self.top_k_results).results()
        except Exception as exc:
            return f"Arxiv exception: {exc}"
        docs = [
            f'Published: {result.updated.date()}\nTitle: {result.title}\n'
            f'Authors: {", ".join(a.name for a in result.authors)}\n'
            f'Summary: {result.summary}'
            for result in results
        ]
        if docs:
            return {'content': '\n\n'.join(docs)}
        return {'content': 'No good Arxiv Result was found'}

if __name__ == "__main__":
    arxivsearch = ArxivSearch()
    query = "mindsearch"
    result = arxivsearch.reply(query)
    print(result)

