from huggingface_hub.hf_api import HfFolder
import transformers
from typing import List
from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from datasets import Dataset
import torch
import requests
from pymilvus import MilvusClient
from PyPDF2 import PdfReader
from pathlib import Path
import pandas as pd
import nltk
nltk.download('punkt')
transformers.logging.set_verbosity_error()

# log in to huggingface
# i know this is a bad practice but for this test
# i think this is acceptable.
HfFolder.save_token('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')


def extract_pdf_text(pdf_url, start_page, end_page, max_words=30):
    # Download the PDF file
    response = requests.get(pdf_url)
    pdf_file = Path("downloaded_file.pdf")
    pdf_file.write_bytes(response.content)

    def split_text(text, max_words=max_words):
        words = nltk.word_tokenize(text)
        result = []

        for i in range(0, len(words), max_words):
            result.append(' '.join(words[i:i + max_words]))

        return result

    # Extract text from the specified pages
    reader = PdfReader(str(pdf_file))
    text = ""
    for page_num in range(start_page - 1, end_page):
        text += reader.pages[page_num].extract_text()

    # Split the text into sentences with 30 words each
    sentences = split_text(text)

    # Write the sentences to a text file as a backup
    with open("extracted_text.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(sentences))

    # Remove the downloaded PDF file
    pdf_file.unlink()

    return sentences


def encode_text(batch):
    # Tokenize sentences
    encoded_input = sentence_tokenizer(
        batch["text"], padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = sentence_model(**encoded_input)

    # Perform pooling
    token_embeddings = model_output[0]
    attention_mask = encoded_input["attention_mask"]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sentence_embeddings = torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Normalize embeddings
    batch["text_embedding"] = torch.nn.functional.normalize(
        sentence_embeddings, p=2, dim=1
    )
    return batch


def extract_context(results: List[dict], threshold: float):
    """ from given list of dictionaries extract the text of each entry if the score is above the threshold
    """
    results = results[0]
    context = ""
    for r in results:
        if r["distance"] > threshold:
            context += r["entity"]["text"] + "\n"
    if context == "":
        context = "answer by saying couldn't find what you are looking for."
    return context


def get_context(milvus_client, collection_name: str, query: str, threshold: float = 0.4, limit=5, output_fields=["text"]):
    query = {"text": [query]}
    query_embedding = [v.tolist()
                       for v in encode_text(query)["text_embedding"]]
    search_results = milvus_client.search(
        collection_name=collection_name,
        data=query_embedding,
        limit=limit,
        output_fields=output_fields,
    )
    return extract_context(search_results, threshold)


def get_formatted_input(messages, context):
    system = """System: This is a chat between a user and an artificial intelligence assistant.
    The assistant gives helpful, detailed, and polite answers to the user's questions based on the context.
    The assistant should also indicate when the answer cannot be found in the context."""
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            # only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] ==
                               "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation

    return formatted_input


def generate_answer(milvus_client, COLLECTION_NAME, user_question) -> str:
    context = get_context(milvus_client, COLLECTION_NAME, user_question)
    messages = [
        {"role": "user", "content": f"{user_question}"}
    ]

    # toggle this to set the context to nothing.
    # context = ""
    formatted_input = get_formatted_input(messages, context)
    tokenized_prompt = gen_tokenizer(
        gen_tokenizer.bos_token + formatted_input, return_tensors="pt").to(gen_model.device)

    terminators = [
        gen_tokenizer.eos_token_id,
        gen_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = gen_model.generate(input_ids=tokenized_prompt.input_ids,
                                 attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators)

    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    answer = gen_tokenizer.decode(response, skip_special_tokens=True)
    final_answer = f"""
    # Question:
    {user_question}
    =================================================================================================
    # Context:
    {context}
    =================================================================================================
    # Answer:
    {answer}
    """
    return final_answer


# Extract text from pdf
pdf_url = "https://assets.openstax.org/oscms-prodcms/media/documents/ConceptsofBiology-WEB.pdf"
# extracting text for chapter 4 to 5
start_page = 103
end_page = 146
extracted_sentences = extract_pdf_text(pdf_url, start_page, end_page)
print(f"Extracted {len(extracted_sentences)} sentences.")


# model for embedding search
sentence_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INFERENCE_BATCH_SIZE = 64  # Batch size of model inference
# Load tokenizer & model from HuggingFace Hub
sentence_tokenizer = AutoTokenizer.from_pretrained(sentence_MODEL)
sentence_model = AutoModel.from_pretrained(sentence_MODEL)

# create a dataset for huggingface model from the list sentences
rag_dataset = Dataset.from_pandas(
    pd.DataFrame(extracted_sentences, columns=["text"]))

rag_dataset = rag_dataset.map(
    encode_text, batched=True, batch_size=INFERENCE_BATCH_SIZE)
rag_data_list = rag_dataset.to_list()

# Connection URI. this will write to a file in the current directory
MILVUS_URI = "/asset_data/milvus.db"
# Collection name
COLLECTION_NAME = "biology_book"
# Embedding dimension. depending on model
DIMENSION = 384

milvus_client = MilvusClient(MILVUS_URI)
if milvus_client.has_collection(collection_name=COLLECTION_NAME):
    milvus_client.drop_collection(collection_name=COLLECTION_NAME)
milvus_client.create_collection(
    collection_name=COLLECTION_NAME,
    dimension=DIMENSION,
    auto_id=True,  # Enable auto id
    enable_dynamic_field=True,  # Enable dynamic fields
    # Map vector field name and embedding column in dataset
    vector_field_name="text_embedding",
    consistency_level="Strong",
)

# add entries to Milvus
milvus_client.insert(collection_name=COLLECTION_NAME, data=rag_data_list)

gen_model_name = "nvidia/Llama3-ChatQA-1.5-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_tokenizer.pad_token_id = gen_tokenizer.eos_token_id
gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_name, quantization_config=bnb_config)
