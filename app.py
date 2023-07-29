import json
from sentence_transformers import SentenceTransformer, util


class InferlessPythonModel:
    def initialize(self):
        self.pipe = SentenceTransformer("sentence-transformers/paraphrase-albert-base-v2")

    def infer(self, inputs):
        query = inputs["query"]
        sentences = inputs["sentences"]

        query_embedding = self.pipe.encode(query)
        sentence_embeddings = []
        for sentence in sentences:
            sentence_embeddings.append(self.pipe.encode(sentence))

        result = {}
        for i in range(len(sentence_embeddings)):
            value = util.cos_sim(query_embedding, sentence_embeddings[i]).item()
            result[sentences[i]] = round(value, 4)

        return {"result": result}

    def finalize(self, args):
        self.pipe = None
