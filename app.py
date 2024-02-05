# Import necessary libraries and modules
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import chainlit as cl
from chainlit.playground.providers import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# Additional imports for language chain functionality
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Pinecone

# Import necessary functions from external libraries
from operator import itemgetter
import pinecone

# =============================================================================
# Retrieval Chain
# =============================================================================

# Function to load the language model
def load_llm():
    # Initialize and return the ChatOpenAI language model
    llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.0,
    )
    return llm

# Function to load the vector store for embeddings
def load_vectorstore():
    # Initialize Pinecone, the vector index, and other necessary components
    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )
    index = pinecone.Index("youtube-index")
    store = LocalFileStore("./cache/")
    core_embeddings_model = OpenAIEmbeddings()

    # Create an embedder with cache-backed embeddings
    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model,
        store,
        namespace=core_embeddings_model.model
    )

    text_field = "text"

    # Create and return a Pinecone vector store
    vectorstore = Pinecone(
        index,
        embedder,
        text_field
    )
    return vectorstore

# Function to define the question-answering chain
def qa_chain():
    # Load the vector store and language model
    vectorstore = load_vectorstore()
    llm = load_llm()
    retriever = vectorstore.as_retriever()

    # Define the template for the question-answering prompt
    template = """You are a helpful assistant that answers questions on the provided context, if its not answered within the context respond with "This query is not directly mentioned by AI Makerspace" then respond to the best of your ability. 
                  Additionally, the context includes a specific integer formatted as <int>, representing a timestamp. 
                  In your response, include this integer as a citation, formatted as a YouTube video link: "https://www.youtube.com/watch?v=[video_id]&t=<int>s" and have the hyperlink text be the title of video.

    ### CONTEXT
    {context}

    ### QUESTION
    {question}
    """

    # Create a ChatPromptTemplate from the defined template
    prompt = ChatPromptTemplate.from_template(template)

    # Define the retrieval-augmented question-answering chain
    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever,
         "question": itemgetter("question")
         }
        | RunnablePassthrough.assign(
            context=itemgetter("context")
        )
        | {
            "response": prompt | llm,
            "context": itemgetter("context"),
        }
    )

    return retrieval_augmented_qa_chain

# =============================================================================
# Chainlit
# =============================================================================

# Define an event handler for when the chat starts
@cl.on_chat_start
async def on_chat_start():
    # Initialize and set the question-answering chain in the user session
    chain = qa_chain()
    cl.user_session.set("chain", chain)
    msg = cl.Message(content="What is your question about AI Makerspace?")
    await msg.send()

# Define an event handler for incoming messages
@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the question-answering chain from the user session
    chain = cl.user_session.get("chain")

    # Invoke the chain with the user's question and retrieve the response
    res = chain.invoke({"question": message.content})

    # Extract the answer from the response and send it as a message
    answer = res['response'].content
    await cl.Message(content=answer).send()

    # Uncomment the following block to show all source documents used
    '''
    source_documents = set()

    for document in res['context']:
        source_url = document.metadata['source_document']
        source_documents.add(source_url)

    combined_message = answer + "\n\nSource Documents:\n" + "\n".join(source_documents)

    await cl.Message(content=combined_message).send()
    '''
