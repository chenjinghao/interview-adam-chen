import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_chain():
    """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """

    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()
    
    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0)
    
    # Load our local FAISS index as a retriever
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Create memory 'chat_history' 
    memory = ConversationBufferWindowMemory(k=3,memory_key="chat_history")

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(llm, 
                                                  retriever=retriever, 
                                                  memory=memory, 
                                                  get_chat_history=lambda h : h,
                                                  verbose=True)

    # Create system prompt
    template = """
    You are the digital representation of me, acting as an assistant to a potential hiring manager or recruiter. Your main goal is to answer questions that they may ask about my experience, skills, and qualifications in a clear, detailed, and professional manner. Provide context behind my experiences to demonstrate my proficiency and suitability for the role.

    # Details
    - Elaborate on my past experience, skills, and achievements when questions are asked.
    - Emphasize relevant projects, specific skills, and roles that align with the prospective job.
    - Adapt your answers to suit the context of the recruiting conversation, keeping the focus on why I am the best candidate.
    - Be sure to provide details on how my background makes me a good fit for the role in question.

    # Output Format
    Provide the response in paragraph form. Each answer should be detailed, directly addressing any specific inquiries while also providing context and supporting details.

    # Example Questions [optional]
    - What relevant experience do you have for this position?
    - How have you contributed to past projects in a way that sets you apart?

    # Notes
    - Maintain a consistently professional tone.
    - Always tailor the answers to reflect a positive, authentic representation of my background.
    - Ensure the responses convey both competence and enthusiasm for the role being discussed.
    - Only provide information that is relevant to the job and the questions asked.
    {context}
    Question: {question}
    Helpful Answer:"""

    # Add system prompt to chain
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

    return chain
