from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from mdc import MDC
import csv
from tqdm import tqdm
from logconf import log_setup
import logging
from typing import Literal, Any, get_args
import argparse
from openai import OpenAI, BadRequestError
import datasets
from datasets import Dataset, concatenate_datasets
import pyarrow as pa
import json
import PyPDF2
import random
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from client_utils import build_openai_client, build_groq_client,build_langchain_embeddings, UsageStats, ChatCompleter, LocalLanguageModel
from math import ceil
from format import DatasetConverter, datasetFormats, outputDatasetTypes
from pathlib import Path
from dotenv import load_dotenv
from checkpointing import Checkpointing, checkpointed
import uuid
import shutil
from threading import Thread, Event
from typing import Optional

log_setup()

load_dotenv()  # take environment variables from .env.

logger = logging.getLogger("raft")

DocType = Literal["pdf", "json", "txt"]
docTypes = list(get_args(DocType))

SystemPromptKey = Literal["gpt", "llama"]
systemPromptKeys = list(get_args(SystemPromptKey))

def get_args() -> argparse.Namespace:
    """
    Parses and returns the arguments specified by the user's command
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--datapath", type=Path, default="", help="If a file, the path at which the document is located. If a folder, the path at which to load all documents")
    parser.add_argument("--output", type=str, default="./", help="The path at which to save the dataset")
    parser.add_argument("--output-format", type=str, default="hf", help="The format of the output dataset.", choices=datasetFormats)
    parser.add_argument("--output-type", type=str, default="jsonl", help="Type to export the dataset to. Defaults to jsonl.", choices=outputDatasetTypes)
    parser.add_argument("--output-chat-system-prompt", type=str, help="The system prompt to use when the output format is chat")
    parser.add_argument("--output-completion-prompt-column", type=str, default="prompt", help="The prompt column name to use for the completion format")
    parser.add_argument("--output-completion-completion-column", type=str, default="completion", help="The completion column name to use for the completion format")
    parser.add_argument("--distractors", type=int, default=3, help="The number of distractor documents to include per data point / triplet")
    parser.add_argument("--p", type=float, default=1.0, help="The percentage that the oracle document is included in the context")
    parser.add_argument("--questions", type=int, default=1, help="The number of data points / triplets to generate per chunk")
    parser.add_argument("--chunk_size", type=int, default=512, help="The size of each chunk in number of tokens")
    parser.add_argument("--min_chunk_length", type=int, default=200, help="The minimum length of a chunk to be included")
    parser.add_argument("--max_chunk_length", type=int, default=900, help="The maximum length of a chunk to be included")
    parser.add_argument("--doctype", type=str, default="txt", help="The type of the document, must be one of the accepted doctypes", choices=docTypes)
    parser.add_argument("--openai_key", type=str, default=None, help="Your OpenAI key used to make queries to GPT-3.5 or GPT-4")
    parser.add_argument("--groq_key", type=str, default=None, help="Your Groq API key used to make queries to Llama models")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small", help="The embedding model to use to encode documents chunks (text-embedding-3-small, ...)")
    parser.add_argument("--completion_model", type=str, default="gpt-4o-mini", help="The model to use to generate questions and answers gpt-4o-mini,llama-3.1-70b-versatile...)")
    parser.add_argument("--system-prompt-key", default="gpt", help="The system prompt to use to generate the dataset", choices=systemPromptKeys)
    parser.add_argument("--workers", type=int, default=2, help="The number of worker threads to use to generate the dataset")
    parser.add_argument("--auto-clean-checkpoints", type=bool, default=False, help="Whether to auto clean the checkpoints after the dataset is generated")
    parser.add_argument("--qa-threshold", type=int, default=None, help="The number of Q/A samples to generate after which to stop the generation process. Defaults to None, which means generating Q/A samples for all documents")

    args = parser.parse_args()
    return args


def get_chunks(
    data_path: Path, 
    doctype: DocType = "pdf", 
    chunk_size: int = 512, 
    openai_key: Optional[str] = None,
    model: str = None,
    min_chunk_length: int = 200,
    max_chunk_length: int = 900
) -> list[str]:
    """
    Takes in a `data_path` and `doctype`, retrieves the document, breaks it down into chunks of size
    `chunk_size`, and returns the chunks.
    """
    chunks = []

    logger.info(f"Retrieving chunks from {data_path} of type {doctype} using the {model} model.")

    embeddings = build_langchain_embeddings(openai_api_key=openai_key, model=model)
    chunks = []
    file_paths = [data_path]
    if data_path.is_dir():
        file_paths = list(data_path.rglob('**/*.' + doctype))

    futures = []
    with tqdm(total=len(file_paths), desc="Chunking", unit="file") as pbar:
        with ThreadPoolExecutor(max_workers=2) as executor:
            for file_path in file_paths:
                futures.append(executor.submit(get_doc_chunks, embeddings, file_path, doctype, chunk_size,min_chunk_length))
            for future in as_completed(futures):
                doc_chunks = future.result()
                chunks.extend(doc_chunks)
                pbar.set_postfix({'chunks': len(chunks)})
                pbar.update(1)

    return chunks

def get_doc_chunks(
    embeddings: OpenAIEmbeddings,
    file_path: Path, 
    doctype: DocType = "pdf", 
    chunk_size: int = 512,
    min_chunk_length: int = 200,
    max_chunk_length: int = 900
 ) -> list[str]:
    if doctype == "json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        text = data["text"]
    elif doctype == "pdf":
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()
    elif doctype == "txt":
        with open(file_path, 'r') as file:
            data = file.read()
        text = str(data)
    else:
        raise TypeError("Document is not one of the accepted types: api, pdf, json, txt")
    
    num_chunks = ceil(len(text) / chunk_size)
    logger.debug(f"Splitting text into {num_chunks} chunks.")

    text_splitter = SemanticChunker(embeddings, number_of_chunks=num_chunks)
    chunks = text_splitter.create_documents([text])
    chunks = [chunk.page_content for chunk in chunks if min_chunk_length <= len(chunk.page_content) <= max_chunk_length]

    filtered_count = num_chunks - len(chunks)
    logger.info(f"Got {num_chunks} chunks, filtered {filtered_count} chunks (shorter than {min_chunk_length} or longer than {max_chunk_length} characters)")
    return chunks


build_qa_messages = {
    "gpt": lambda chunk, x : [
            {"role": "system", "content": 
            """You are a professional synthetic question generator.
               Instructions:
               - Given a chunk of context about some topic(s), generate %s different example questions a user could ask in English
               - Questions should be factual and of intermediate difficulty.
               - Questions should be answerable using only information and some analysis of the text from the chunk.
               - Questions should be answerable by a sentence or short phrase, not a single word.
               - Generate one question per line
               - Generate only questions
               - Questions should be succinct
               - Use a variety of question starters (What, How, Why, When, etc.)
              
               Important: Make sure the question is a complete sentence AlWAYS ended with question mark!!
               Here is an example:
               Context: A Wikipedia paragraph about vampire bats
               Question: How do vampire bats' feeding habits differ from those of other bat species?
               """ % (x)},
            {"role": "user", "content": str(chunk)}
        ],
    "llama": lambda chunk, x : [
            {"role": "system", "content": 
            """You are a professional synthetic question generator.
               Instructions:
               - Given a chunk of context about some topic(s), generate %s different example questions a user could ask in English
               - Questions should be factual and of intermediate difficulty and specific.
               - Questions should be answerable using only information and some analysis of the text from the chunk.
               - Questions should be answerable by a sentence or short phrase, not a single word.
               - Generate one question per line, each question is seperated by "\n", dont give a serial number to the questions!!
               - Generate only questions directly after "Real Questions:"!
               - Questions should be succinct
               - Use a variety of question starters (What, How, Why, When, etc.)
              
               Important: Make sure the question is a complete sentence AlWAYS ended with question mark!!
               Here is an example:
               Example Context: A Wikipedia paragraph about vampire bats
               Example Questions: How do vampire bats' feeding habits differ from those of other bat species?\nWhat are the major habitants of vampire bats?
               """ % (x)},
            {"role": "user", "content": "Real Context:"+str(chunk)+"\nReal Questions:"}
        ]
}

def generate_instructions_gen(chat_completer: ChatCompleter, chunk: Any, x: int = 5, model: str = None, prompt_key : str = "gpt") -> list[str]:
    """
    Generates `x` questions / use cases for `chunk`. Used when the input document is of general types 
    `pdf`, `json`, or `txt`.
    """
    try:
        response = chat_completer(
            model=model,
            messages=build_qa_messages[prompt_key](chunk, x),
            temperature= 0.7,
            max_tokens=min(50 * x, 512), # 25 tokens per question
        )
        if response is None:
            logger.warning("Received incomplete response. Returning empty list.")
            return []
    except BadRequestError as e:
        if e.code == "content_filter":
            logger.warning(f"Got content filter error, skipping chunk: {e.message}")
            return []
        raise e

    if "mistral" in model.lower():
        content = response
    else:
        content = response.choices[0].message.content
    queries = content.split('\n') if content else []
    #queries = [strip_str(q) for q in queries]
    #remove numbers for mistral outputs 
    queries = [q.lstrip('0123456789. ').strip() for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]

    return queries

def strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by GPT-4.
    """
    l, r = 0, len(s)-1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i 
    r += 2
    return s[l:min(r, len(s))]


prompt_templates = {
    "gpt": """
        Question: {question}\nContext: {context}\n
    """,
    "llama": """
        Question: {question}\nContext: {context}\n
    """
    }

system_answer_prompt = """
    You are a professional question answering expert in IT. Answer the user question below in English using only the given context.

    Instructions:
       1) Provide step-by-step chain-of-thought (CoT) on how to answer the question in the form <CoT>:.
       2) Explain which parts of the context are meaningful and why.
       3) Include a quote from the context, enclosed in ##begin_quote## and ##end_quote##.
       4) Provide a summary of how you reached your answer.
       5) End your response with the final answer in English in the form <ANSWER>: [The original sentences from the context].
   - If the context doesn't provide a clear answer, state this and provide the best possible answer based on the available information.
  
   Here is an example:
   Example question: What movement was generated by Jack Weinberg's arrest in Sproul Plaza?
   Example answer: <CoT>:
   1. Identify key information: The context mentions Jack Weinberg's arrest and its consequences.
   2. Locate relevant quote: ##begin_quote##The arrest in Sproul Plaza of Jack Weinberg, a recent Berkeley alumnus and chair of Campus CORE, prompted a series of student-led acts of formal remonstrance and civil disobedience that ultimately gave rise to the Free Speech Movement##end_quote##.
   3. Analyze the quote: The arrest led to student protests which evolved into a specific movement.
   4. Summary: The context directly links Weinberg's arrest to the rise of the Free Speech Movement.
   <ANSWER>: The arrest in Sproul Plaza of Jack Weinberg, a recent Berkeley alumnus and chair of Campus CORE, prompted a series of student-led acts of formal remonstrance and civil disobedience that ultimately gave rise to the Free Speech Movement
   """

def encode_question_gen(question: str, chunk: Any, prompt_key : str = "gpt") -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
    
    prompts = []

    user_prompt = prompt_templates[prompt_key].format(question=question, context=str(chunk))
    prompts.append({"role": "system", "content": system_answer_prompt})
    prompts.append({"role": "user", "content": user_prompt})
    return prompts

def generate_label(chat_completer: ChatCompleter, question: str, context: Any, doctype: DocType = "pdf", model: str = None, prompt_key : str = "gpt") -> Optional[str]:
    """
    Generates the label / answer to `question` using `context` and GPT-4.
    """
    question = encode_question_gen(question, context, prompt_key)
    response = chat_completer(
        model=model,
        messages=question,
        n=1,
        temperature=0.7,
        max_tokens=512,
    )
    if "mistral" in model.lower():
        return response
    response = response.choices[0].message.content
    return response
    

def generate_question_cot_answer(
        chat_completer: ChatCompleter,
        chunks: list[str], 
        chunk: str, 
        chunk_id, 
        question,
        doctype: DocType = "api", 
        num_distract: int = 3, 
        p: float = 0.8,
        model: str = None,
        prompt_key: str = "gpt",
        ):
    datapt = {
            "id": None,
            "type": None,
            "question": None,
            "context": None,
            "oracle_context": None,
            "cot_answer": None
        }

    datapt["id"] = str(uuid.uuid4())
    datapt["type"] = "general"
    datapt["question"] = question

    # add num_distract distractor docs
    docs = [chunk]
    indices = list(range(0, len(chunks)))
    indices.remove(chunk_id)

    if len(indices) <= num_distract:
        for j in indices:
            docs.append(chunks[j])
    else:
        for j in random.sample(indices, num_distract):
            docs.append(chunks[j])

    # decides whether to add oracle document
    oracle = random.uniform(0, 1) < p
    if not oracle:
        docs[0] = chunks[random.sample(indices, 1)[0]]
    random.shuffle(docs)

    d = {
        "title": [],
        "sentences": []
    }

    d["title"].append(["placeholder_title"]*(num_distract+1))
    d["sentences"].append(docs)
    datapt["context"] = d
    datapt["oracle_context"] = chunk

    # add answer to q
    datapt["cot_answer"] = generate_label(chat_completer, question, chunk, doctype, model=model, prompt_key=prompt_key)

    # construct model instruction 
    context = ""
    for doc in docs:
        context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
    context += question
    datapt["instruction"] = context
    return datapt

def build_or_load_chunks(
        datapath: Path, 
        doctype: str,
        CHUNK_SIZE: int, 
        OPENAPI_API_KEY: str,
        embedding_model: str,
        checkpoints_dir: Path, 
        min_chunk_length: int,
        max_chunk_length: int
        ):
    """
    Builds chunks and checkpoints them if asked
    """
    chunks_ds: Dataset = None
    chunks = None
    checkpoints_chunks_path = checkpoints_dir / "chunks"
    logger.info(f"Using checkpoint chunks {checkpoints_chunks_path}")
    if checkpoints_chunks_path.exists():
        chunks_ds = Dataset.load_from_disk(checkpoints_chunks_path)
        chunks = chunks_ds['chunk']

    if not chunks:
        chunks = get_chunks(datapath, doctype, CHUNK_SIZE, OPENAPI_API_KEY, model=embedding_model,min_chunk_length=min_chunk_length, max_chunk_length=max_chunk_length)

    if not chunks_ds:
        chunks_table = pa.table({ "chunk": chunks })
        chunks_ds = Dataset(chunks_table)
        chunks_ds.save_to_disk(checkpoints_chunks_path)
    return chunks

def main():

    main_start = time.time()

    # run code
    args = get_args()
    completion_model = args.completion_model

    # Validate arguments
    if args.output_chat_system_prompt and args.output_format != "chat":
        raise Exception("Parameter --output-chat-system-prompt can only be used with --output-format chat")

    OPENAPI_API_KEY = args.openai_key
    GROQ_API_KEY = args.groq_key
    openai_client = build_openai_client(env_prefix="COMPLETION", api_key=OPENAPI_API_KEY)

    groq_client = None
    if GROQ_API_KEY:
        groq_client = build_groq_client(env_prefix="GROQ", api_key=GROQ_API_KEY)
    else:
        logger.info("No Groq API key provided. Groq client will not be initialized.")

    mistral_client = None
    if "mistral" in completion_model.lower(): 
        mistral_client = LocalLanguageModel(completion_model)
        logger.info(f"Initialized mistral client with model: {completion_model}")


    chat_completer = ChatCompleter(openai_client, groq_client,mistral_client)

    CHUNK_SIZE = args.chunk_size
    NUM_DISTRACT_DOCS = args.distractors
    #This is where the jsonl file saved 
    output_path = Path(args.output).absolute()

    checkpoints_dir = Path(str(output_path) + "-checkpoints").absolute()
    auto_clean_checkpoints = args.auto_clean_checkpoints
    if auto_clean_checkpoints:
        logger.info(f"Checkpoints will be automatically deleted after dataset generation. Remove --auto-clean-checkpoints to deactivate.")

    datapath: Path = args.datapath

    datasets.disable_progress_bars()

    # Chunks
    chunks = build_or_load_chunks(datapath, args.doctype, CHUNK_SIZE, OPENAPI_API_KEY, args.embedding_model, checkpoints_dir,args.min_chunk_length,args.max_chunk_length)

    cot_answers_ds = None

    num_chunks = len(chunks)
    num_questions = args.questions
    max_workers = args.workers
    doctype = args.doctype
    completion_model = args.completion_model

    system_prompt_key = args.system_prompt_key

    logger.info(f"Using system prompt key {system_prompt_key}")
    logger.info(f"Using completion model {completion_model}")

    logger.info(f"Using {max_workers} worker threads")

    cot_answers_ds = stage_generate(chat_completer, checkpoints_dir, chunks, num_questions, max_workers, doctype, completion_model, system_prompt_key, num_distract=NUM_DISTRACT_DOCS, p=args.p, qa_threshold=args.qa_threshold)

    # Save as .arrow format
    datasets.enable_progress_bars()
    cot_answers_ds.save_to_disk(str(output_path))

    # Save as .jsonl format
    formatter = DatasetConverter()

    # Extract format specific params
    format_params = {}
    if args.output_chat_system_prompt:
        format_params['system_prompt'] = args.output_chat_system_prompt

    if args.output_format == "completion":
        format_params['prompt_column'] = args.output_completion_prompt_column
        format_params['completion_column'] = args.output_completion_completion_column

    #prepare csv output path
    jsonl_file_path = output_path.with_suffix('.jsonl')
    csv_output_path = Path(str(output_path).replace('raft_data_En', 'csvdata_En')).with_suffix('.csv')

    formatter.convert(ds=cot_answers_ds, format=args.output_format, output_path=str(output_path), output_type=args.output_type, params=format_params)

    # Warning, this deletes all intermediary checkpoint files
    if auto_clean_checkpoints:
        shutil.rmtree(checkpoints_dir)

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    create_csv_from_jsonl(jsonl_file_path, csv_output_path)

    logger.info(f"Generated {len(cot_answers_ds)} question/answer/CoT/documents samples")
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"CSV dataset saved to {csv_output_path}")
    logger.info(f"Done in {time.time() - main_start:.2f}s")

class StoppingException(Exception):
    """
    Raised by worker threads when the process is stopping early
    """
    pass

def stage_generate(chat_completer: ChatCompleter, checkpoints_dir, chunks, num_questions, max_workers, doctype, completion_model, system_prompt_key, num_distract, p, qa_threshold):
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """

    questions_checkpointing = Checkpointing(checkpoints_dir / "questions")
    answers_checkpointing = Checkpointing(checkpoints_dir / "answers")
    num_chunks = len(chunks)

    # Tracking when the process is stopping, so we can stop the generation process early
    # Initial value is False
    is_stopping = Event()

    @checkpointed(questions_checkpointing)
    def generate_chunk_instructions_ds(chunk: str, chunk_id: int, doctype: str, *args, **kwargs):
        """
        Generates a dataset of instructions for a given chunk.
        """
        questions = generate_instructions_gen(chunk=chunk, *args, **kwargs)
        chunk_question_pairs = [{"chunk": chunk, "chunk_id": chunk_id, "question": question} for question in questions]
        questions_ds = Dataset.from_list(chunk_question_pairs)
        return questions_ds

    @checkpointed(answers_checkpointing)
    def generate_question_cot_answers(questions_ds, chunk_id: int, chunk: str, *args, **kwargs):
        def process_example(chunk, question):
            try:
                cot_answer = generate_question_cot_answer(chunk=chunk, chunk_id=chunk_id, chunks=chunks, question=question, *args, **kwargs)
            except BadRequestError as e:
                if e.code == "content_filter":
                    logger.warning(f"Got content filter error, skipping question '{question}': {e.message}")
                    return None
                raise e

            return cot_answer

        results = [process_example(chunk, question) for chunk, question in zip(questions_ds['chunk'], questions_ds['question'])] if len(questions_ds) > 0 else []
        results = [r for r in results if r is not None]
        table = pa.Table.from_pylist(results)
        ds = Dataset(table)
        return ds

    def process_chunk(i):
        if is_stopping.is_set():
            raise StoppingException()
        chunk = chunks[i]
        questions_ds = generate_chunk_instructions_ds(chunk=chunk, chunk_id=i, chat_completer=chat_completer, x=num_questions, model=completion_model, doctype=doctype, prompt_key=system_prompt_key)
        answers_ds = generate_question_cot_answers(questions_ds=questions_ds, chunk=chunk, chunk_id=i, chat_completer=chat_completer, model=completion_model, doctype=doctype, prompt_key=system_prompt_key, num_distract=num_distract, p=p)
        return answers_ds

    futures = []
    answers_ds_list = []
    usage_stats = UsageStats()

    # we use the checkpointing to keep track of the chunks that have already been processed
    # the answers are generated after the questions so the process might have been stopped in between a batch of answers and matching questions
    # so we need to use the answers checkpointing to keep track of which chunks we need to process
    # if the questions for a given chunk have already been checkpointed, they will just be loaded from the checkpoint
    # we set the tqdm's initial position to avoid having cached data skew the stats
    missing_chunks = answers_checkpointing.missing_checkpoints(num_chunks)

    gen_questions_count = 0
    if answers_checkpointing.has_checkpoints():
        ds = answers_checkpointing.collect_checkpoints()
        gen_questions_count = len(ds)

    done_chunks = num_chunks - len(missing_chunks)
    if done_chunks > 0 or gen_questions_count > 0:
        logger.info(f"Resuming generation from chunk {done_chunks}/{num_chunks} and {gen_questions_count} questions")

    # If we have a QA threshold, it makes more sense to keep track of the number of questions generated
    # Otherwise, track chunks
    track_questions = qa_threshold is not None

    if qa_threshold:
        logger.info(f"Will stop early as soon as the QA threshold is met: {qa_threshold}")

    if track_questions:
        tqdm_args = {"total": qa_threshold, "unit": "qa", "initial": gen_questions_count}
    else:
        tqdm_args = {"total": num_chunks, "unit": "chunk", "initial": done_chunks}

    tps = 0
    with tqdm(desc="Generating", **tqdm_args) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in missing_chunks:
                futures.append(executor.submit(process_chunk, i))
            for future in as_completed(futures):
                if qa_threshold and gen_questions_count >= qa_threshold:
                    logger.info(f"Met threshold {gen_questions_count} >= {qa_threshold} questions, stopping generation")
                    is_stopping.set()
                    break
                answers_ds = future.result()
                answers_ds_list.append(answers_ds)
                increment = min(len(answers_ds), qa_threshold - gen_questions_count) if track_questions else 1
                gen_questions_count += len(answers_ds)
                done_chunks += 1
                stats = chat_completer.get_stats_and_reset()
                if stats:
                    tps = stats.total_tokens / stats.duration
                    usage_stats += stats
                postfix = {'last tok/s': tps, 'avg tok/s': usage_stats.total_tokens / usage_stats.duration if usage_stats.duration > 0 else 0}
                if track_questions:
                    postfix['chunks'] = done_chunks
                else:
                    postfix['qa'] = gen_questions_count
                pbar.set_postfix(postfix)
                pbar.update(increment)

    ds = answers_checkpointing.collect_checkpoints()
    ds = ds.select(range(qa_threshold)) if qa_threshold else ds
    logger.info(f"Consumed {usage_stats.prompt_tokens} prompt tokens, {usage_stats.completion_tokens} completion tokens, {usage_stats.total_tokens} total tokens")

    return ds

def create_csv_from_jsonl(jsonl_path, csv_path):
    with jsonl_path.open('r') as json_file, csv_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['id', 'question', 'cot_answer', "oracle_context", 'context'])
        writer.writeheader()

        lines = json_file.readlines()
        random.shuffle(lines)

        for line in lines:
            data = json.loads(line.strip())
            # Write each record to CSV
            writer.writerow({
                'id': data['id'],
                'question': data['question'],
                'cot_answer': data['cot_answer'],
                "oracle_context": data["oracle_context"],
                'context': data['context']['sentences'][0] 
            })

    logger.info("CSV file created successfully.")

if __name__ == "__main__":
    with MDC(progress="0%"):
        main()
