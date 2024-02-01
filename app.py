#inports
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

import chainlit as cl  
from chainlit.playground.providers import ChatOpenAI  
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Pinecone

from operator import itemgetter
import pinecone

# =============================================================================
# Retrieval Chain
# =============================================================================
def load_llm():
  llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.0,
    )
  return llm


def load_vectorstore():

    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_ENV')
    )

    index = pinecone.Index("youtube-index")
    store = LocalFileStore("./cache/")
    core_embeddings_model = OpenAIEmbeddings()

    embedder = CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model,
        store,
        namespace=core_embeddings_model.model
    )

    text_field = "text"

    vectorstore = Pinecone(
        index,
        embedder,  
        text_field
    )

    return vectorstore


def qa_chain():

    vectorstore = load_vectorstore()
    llm = load_llm()
    retriever = vectorstore.as_retriever()

    template = """You are a helpful assistant that answers questions on the provided context, if its not answered within the context respond with "This query is not directly mentioned by AI Makerspace" then respond to the best of your ability. 
                  Additionally, the context includes a specific integer formatted as <int>, representing a timestamp. 
                  In your response, include this integer as a citation, formatted as a YouTube video link: "https://www.youtube.com/watch?v=[video_id]&t=<int>s" and have the hyperlink text be the title of video.


    ### CONTEXT
    {context}

    ### QUESTION
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever,
        "question": itemgetter("question")
        }
        | RunnablePassthrough.assign(
            context=itemgetter("context")
        )
        | {
            "response": prompt  | llm,
            "context": itemgetter("context"),
        }
    )

    return retrieval_augmented_qa_chain

# =============================================================================
# Chainlit
# =============================================================================
@cl.on_chat_start
async def on_chat_start():
    chain = qa_chain()
    cl.user_session.set("chain", chain)
    msg=cl.Message(content="What is your question about AI Makerspace?")
    await msg.send()

@cl.on_message
async def on_message(message: cl.Message):
    chain=cl.user_session.get("chain")
    res = chain.invoke({"question" : message.content})

    answer = res['response'].content
    await cl.Message(content=answer).send()
    
    #Use to show all source documents used
    '''
    source_documents = set()

    for document in res['context']:
        source_url = document.metadata['source_document']
        source_documents.add(source_url)

    combined_message = answer + "\n\nSource Documents:\n" + "\n".join(source_documents)

    await cl.Message(content=combined_message).send()
    '''
