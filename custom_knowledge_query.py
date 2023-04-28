from langchain import PromptTemplate, LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores.faiss import FAISS

# SCRIPT INFO:
# 
# This script allows you to create a vectorstore from a file and query it with a question (hard coded).
# 
# It shows how you could send questions to a GPT4All custom knowledge base and receive answers.
# 
# If you want a chat style interface using a similar custom knowledge base, you can use the custom_chatbot.py script provided.

# Setup 
gpt4all_path = './models/gpt4all-converted.bin' 
llama_path = './models/ggml-model-q4_0.bin' 

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
loader = TextLoader('./docs/shortened_sotu.txt')
embeddings = LlamaCppEmbeddings(model_path=llama_path)
llm = GPT4All(model=gpt4all_path, callback_manager=callback_manager, verbose=True)

# Split text 
def split_chunks(sources):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    for chunk in splitter.split_documents(sources):
        chunks.append(chunk)
    return chunks


def create_index(chunks):
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    search_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

    return search_index


def similarity_search(query, index):
    matched_docs = index.similarity_search(query, k=4)
    sources = []
    for doc in matched_docs:
        sources.append(
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        )

    return matched_docs, sources


# Create Index
# docs = loader.load()
# chunks = split_chunks(docs)
# index = create_index(chunks)

# Save Index (use this to save the index for later use)
# Comment the line below after running once successfully (IMPORTANT)

# index.save_local("state_of_the_union_index")

# Load Index (use this to load the index from a file, eg on your second time running things and beyond)
# Uncomment the line below after running once successfully (IMPORTANT)

index = FAISS.load_local("./full_sotu_index", embeddings)

# Set your query here manually
question = "Summarize the comments about NATO and its purpose."
matched_docs, sources = similarity_search(question, index)

template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step."""

context = "\n".join([doc.page_content for doc in matched_docs])
prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))