import json
from sentence_transformers import SentenceTransformer, util


class InferlessPythonModel:
    def initialize(self):
        self.pipe = SentenceTransformer("sentence-transformers/paraphrase-albert-base-v2")

    def infer(self, inputs):
        sentence_1 = inputs["sentence_1"]
        sentence_2 = inputs["sentence_2"]

        embeddings = self.pipe.encode([sentence_1, sentence_2])

        return {"result": str(embeddings.tolist())}

    def finalize(self, args):
        self.pipe = None
