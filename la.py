import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import json
from openai import OpenAI
from simple_logger import SimpleLogger


class RetrievalAugmentedGenerationTest:
    """ Exploring task... 
    Don't even think of running this on Python lower than 3.10.7, 
    although it might just work fine on Python down-to 3.7, but no guaranties.

    TODO: Too many things to enumerate now...
    """

    STM = 'all-MiniLM-L6-v2'

    def __init__(self) -> None:
        self.logger = SimpleLogger(logfile='rag.log')
        self.logger.debug('Initializing, things are starting to happen, so please be patient...')

        self.open_ai_api = "http://localhost:8080/v1/"
        self.open_ai_api_key = "sk-no-key-required"

        # To be able to map sentences & paragraphs to a 384 dimensional dense vector
        # space, can be used for tasks like clustering or semantic search.
        self.logger.debug(f'Loading Sentence Transformer Model: {self.STM}...')
        self.mapper = SentenceTransformer(self.STM)

        self.logger.debug('Qdrant DB...')
        # self.qdrant = QdrantClient(":memory:") 
        self.qdrant = QdrantClient("localhost", port=6333)

        self.logger.debug('Done Initializing.')

    # def mapper_self_check(self):
    #     self.logger.debug('Just a quick sanity check to see if SentenceTransformer mapper works...')
    #     sentences = ["This is an example sentence", "Each sentence is converted"]
    #     embeddings = self.mapper.encode(sentences)
    #     self.logger.debug(f'Mapper embeddings results: {embeddings}')

    def get_csv_data(self, csv_file, sample=None):
        self.logger.debug(f'Loading CSV file: "{csv_file}" ...')
        df = pd.read_csv(csv_file)
        # If you need remove any NaN values which blows up serialization, uncomment... 
        # Well, if you have perfect CSV files without any of
        # NaN/Null/None/Empty/etc data, than you don't need to...
        df = df[df['variety'].notna()]
        if sample:
            data = df.sample(sample).to_dict('records')
        data = df.to_dict('records')
        return data

    def create_collection(self, collection_name):
        # Distance.COSINE measures the cosine of the angle between two vectors. 
        # For word embeddings, the vectors represent the semantic meaning of 
        # words in a high-dimensional space. Cosine similarity ranges from -1 to 1
        self.logger.debug(f'Creating Qdrant collection: "{collection_name}" ...')
        self.qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                # Vector size is defined by used mapper/model
                size=self.mapper.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )

    def vectorize_collection(self, collection_name, field, data):
        self.logger.debug(
            f'Vectorizing Qdrant collection: "{collection_name}", '
            f'you may hear the CPU fan a bit louder while doing it...')
        self.qdrant.upload_points(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=self.mapper.encode(doc[field]).tolist(),
                    payload=doc,
                ) for idx, doc in enumerate(data)
            ]
        )

    def query_collection(self, user_prompt, collection_name, limit):
        self.logger.debug(
            f'Querying Qdrant collection"{collection_name}" with "{user_prompt}", limiting max results to {limit}...')
        hits = self.qdrant.search(
            collection_name=collection_name,
            query_vector=self.mapper.encode(user_prompt).tolist(),
            limit=limit
        )
        return [hit.payload for hit in hits]

    def query_open_ai(self, user_prompt, local_search_results):
        # We simulate locally ChatGPT with a local 2GB LLM, phi-2 which has an Open AI compatible API
        # You can start the llm from the llm.cmd file. We could implement a docker client in python to do it for you
        # but would be to fancy for this scope...
        self.logger.debug(f'Querying an Open AI Compatible LLM...')
        client = OpenAI(base_url=self.open_ai_api, api_key=self.open_ai_api_key)

        completion = client.chat.completions.create(
            # model="gpt-4"
            model="phi-2",
            messages=[
                {
                    "role": "system",
                    "content": "Role: Chat-bot named Knut, a dev-ops expert and software professional. "
                               "Icon-Image: Cute Owl who also inspires intelligence and confidence. "
                               "Skills: Analyze build logs. Identify errors and their causes. Suggest potential fixes. "
                               "Priority: Assist users in analyzing software build logs from CI/CD pipelines. "
                               "Focus: Technical topics related to build logs only. "
                               "Exception: If asked repeatedly, offer advice on any subject."
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": str(local_search_results)
                }
            ]
        )
        return completion

    def main(self):

        input_file = 'top_rated_wines.csv'
        max_data_lines = 700

        collection_name = 'top_wines'
        interesting_field = 'notes'

        user_prompt = "Suggest me an amazing Bordeaux wine from France"

        user_prompt = "Who are you, and what can you do?"

        results_hit_limit = 3

        # self.mapper_self_check()
        data = self.get_csv_data(input_file, max_data_lines)
        self.create_collection(collection_name)
        self.vectorize_collection(collection_name, interesting_field, data)

        local_search_results = self.query_collection(user_prompt, collection_name, results_hit_limit)
        pretty_j = json.dumps(local_search_results, indent=3)
        self.logger.debug(f'Local collection search result is: \n{pretty_j}')

        prompt_response = self.query_open_ai(user_prompt, local_search_results)
        self.logger.debug(f'AI role response is: \n{prompt_response.choices[0].message}')
        self.logger.debug(f'Open AI (Compatible) LLM full response object is: \n{prompt_response}')


if __name__ == '__main__':
    rt = RetrievalAugmentedGenerationTest()
    rt.main()
