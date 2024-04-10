import redis
import csv
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import *
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import openai
import tiktoken

from config.logger_config import setup_logger

# Setup logger
logger = setup_logger()

# Load environment variables
load_dotenv()

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379

REDIS_URL = os.getenv('GABS_REDIS_URL', "redis://localhost:6379")
OPENAI_API_KEY = os.getenv('GABS_OPENAI_API_KEY', 'sk_nadanadanada')

VSS_INDEX_TYPE = "HNSW"
VSS_DATA_TYPE = "FLOAT32"
VSS_DISTANCE = "COSINE"
VSS_DIMENSION = 384
VSS_MINIMUM_SCORE = 2

MAX_MOVIES = 50000

conn = redis.Redis.from_url(REDIS_URL, decode_responses=True)
print(conn.ping())


def load():
    with open("data/movies/imdb_movies.csv", encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)
        cnt = 0
        for row in csvReader:
            conn.json().set(f'moviebot:movie:{cnt}', '$', row)
            cnt = cnt + 1
            if (cnt > MAX_MOVIES):
                break
        print("Data was loaded")


def create_index():
    indexes = conn.execute_command("FT._LIST")
    if "movie_idx" not in indexes:
        index_def = IndexDefinition(prefix=["moviebot:movie:"], index_type=IndexType.JSON)
        schema = (TextField("$.crew", as_name="crew"),
                  TextField("$.overview", as_name="overview"),
                  TagField("$.genre", as_name="genre"),
                  TagField("$.names", as_name="names"),
                  VectorField("$.overview_embedding", VSS_INDEX_TYPE,
                              {"TYPE": VSS_DATA_TYPE, "DIM": VSS_DIMENSION, "DISTANCE_METRIC": VSS_DISTANCE},
                              as_name="embedding"))
        conn.ft('movie_idx').create_index(schema, definition=index_def)
        print("The index has been created")
    else:
        print("The index exists")


def create_embeddings():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    for key in conn.scan_iter(match='moviebot:movie:*'):
        print(f"creating the embedding for {key}")
        result = conn.json().get(key, "$.names", "$.overview", "$.crew", "$.score", "$.genre")
        movie = f"movie title is: {result['$.names'][0]}\n"
        movie += f"movie genre is: {result['$.genre'][0]}\n"
        movie += f"movie crew is: {result['$.crew'][0]}\n"
        movie += f"movie score is: {result['$.score'][0]}\n"
        movie += f"movie overview is: {result['$.overview'][0]}\n"
        conn.json().set(key, "$.overview_embedding", model.encode(movie).astype(np.float32).tolist())


def get_prompt(model, query):
    context = ""
    prompt = ""
    q = Query("@embedding:[VECTOR_RANGE $radius $vec]=>{$YIELD_DISTANCE_AS: score}") \
        .sort_by("score", asc=True) \
        .return_fields("overview", "names", "score", "$.crew", "$.genre", "$.score") \
        .paging(0, 5) \
        .dialect(2)

    # Find all vectors within VSS_MINIMUM_SCORE of the query vector
    query_params = {
        "radius": VSS_MINIMUM_SCORE,
        "vec": model.encode(query).astype(np.float32).tobytes()
    }

    res = conn.ft("movie_idx").search(q, query_params)

    if (res is not None) and len(res.docs):
        it = iter(res.docs[0:])
        for x in it:
            # print("the score is: " + str(x['score']))
            movie = f"movie title is: {x['names']}\n"
            movie += f"movie genre is: {x['$.genre']}\n"
            movie += f"movie crew is: {x['$.crew']}\n"
            movie += f"movie score is: {x['$.score']}\n"
            movie += f"movie overview is: {x['overview']}\n"
            context += movie + "\n"

    if len(context) > 0:
        prompt = '''Use the provided information to answer the search query the user has sent. The information in the 
        database provides three movies, choose the one or the ones that fit most. If you can't answer the user's 
        question, say "Sorry, I am unable to answer the question, try to refine your question". Do not guess. You 
        must deduce the answer exclusively from the information provided. The answer must be formatted in markdown or 
        HTML. Do not make things up. Do not add personal opinions. Do not add any disclaimer.

            Search query: 

            {}

            Information in the database: 

            {}
            '''.format(query, context)

    return prompt


def getOpenAIGPT35(prompt):
    # Define the system message
    system_msg = ('You are a smart and knowledgeable AI assistant with expertise in all kinds of movies. You are a '
                  'very friendly and helpful AI. You are empowered to recommend movies based on the provided context. '
                  'Do NOT make anything up. Do NOT engage in topics that are not about movies.')

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
    # print("tokens: " + str(num_tokens_from_string(prompt, "cl100k_base")))

    try:
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613",
                                                stream=False,
                                                messages=[{"role": "system", "content": system_msg},
                                                          {"role": "user", "content": prompt}])
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        # Handle the error here
        if "context window is too large" in str(e):
            print("Error: Maximum context length exceeded. Please shorten your input.")
            return "Maximum context length exceeded"
        else:
            print("An unexpected error occurred:", e)
            return "An unexpected error occurred"


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def render():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # React to user input
    while True:
        question = input("Ask a question\n")
        reply = f"You asked: {question}"
        prompt = get_prompt(model, question)
        response = getOpenAIGPT35(prompt)
        print(response)
        print("--------------------------------")


def main():
    """Main function to orchestrate the operations."""
    # Load movies, create index, and generate embeddings as needed
    # Uncomment the following lines if running for the first time or when needed:
    load()
    create_index()
    create_embeddings()

    render()


if __name__ == "__main__":
    main()
