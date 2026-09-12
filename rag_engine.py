import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class ShelfSenseRAG:

    def __init__(self):

        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        knowledge_path = os.path.join(
            base_dir,
            "knowledge_base.json"
        )

        with open(
            knowledge_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.knowledge = json.load(file)

        # Create object lookup
        self.object_lookup = {}

        for item in self.knowledge:

            object_name = (
                item["object"]
                .strip()
                .lower()
            )

            self.object_lookup[object_name] = item

        # Load embedding model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # Create documents
        self.documents = []

        for item in self.knowledge:

            text = (
                f"Object: {item['object']}. "
                f"Category: {item['category']}. "
                f"Description: {item['description']} "
                f"Uses: {item['uses']} "
                f"Related objects: {item['related_objects']} "
                f"Organization: {item['organization']}"
            )

            self.documents.append(text)

        # Create embeddings
        embeddings = self.model.encode(
            self.documents
        )

        embeddings = np.array(
            embeddings
        ).astype("float32")

        # Create FAISS index
        self.index = faiss.IndexFlatL2(
            embeddings.shape[1]
        )

        self.index.add(
            embeddings
        )


    def search(
        self,
        query,
        top_k=3
    ):

        query = (
            query
            .strip()
            .lower()
        )

        # Exact object match
        if query in self.object_lookup:

            return [
                self.object_lookup[query]
            ]

        # Semantic search
        query_embedding = self.model.encode(
            [query]
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for index in indices[0]:

            if (
                0 <= index
                < len(self.knowledge)
            ):

                results.append(
                    self.knowledge[index]
                )

        return results