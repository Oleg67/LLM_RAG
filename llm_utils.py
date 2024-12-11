#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  llm_utils.py
#  
#  Copyright 2024 oleg <oleg@oleg-GL553VD>
#  
#  
#  

import bs4
from langchain_community.document_loaders import SeleniumURLLoader, TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import SeleniumURLLoader, TextLoader
from langchain_text_splitters import (RecursiveCharacterTextSplitter, 
                                      NLTKTextSplitter,
                                      SentenceTransformersTokenTextSplitter
                                     )
from langchain_experimental.text_splitter import SemanticChunker

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from datasets import Dataset 


import nltk
nltk.download('punkt_tab')

def split_documets(docs, 
                      spliter="nltk", 
                      tokens=500, 
                      overlap=0.1,
                      embeedding_model=OpenAIEmbeddings(),
                     ):
    
    overlap=int(overlap*tokens) if overlap>0 else 0
    
    if  spliter == "nltk":
        text_splitter = NLTKTextSplitter(chunk_size=tokens, 
                                         chunk_overlap=overlap)
        
    elif spliter == "sentence":
        text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=tokens,
                                                              model_name ="BAAI/bge-base-en-v1.5",
                                                              chunk_overlap=overlap)
        
    elif spliter == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n"],
                                                       chunk_size=tokens, 
                                                       chunk_overlap=overlap)
        
    elif spliter == "semantic":
        text_splitter = SemanticChunker(embeedding_model,
                            breakpoint_threshold_type = 'percentile', # ['percentile', 'standard_deviation', 'interquartile']
                            breakpoint_threshold_amount=None,
                            number_of_chunks=None) 
                                                       
                                                       
    else:
        print( f"Unknown spliter {spliter}")
    
    #text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n"], chunk_size=500, chunk_overlap=50)
    #text_splitter = NLTKTextSplitter(chunk_size=tokens, chunk_overlap=int(overlap*tokens))
    #text_splitter = SentenceTransformersTokenTextSplitter()
    splits = text_splitter.split_documents(docs)
    
    return splits



def load_url_documets(list_urls, 
                      spliter="nltk", 
                      tokens=500, 
                      overlap=0.1,
                      embeedding_model=OpenAIEmbeddings(),
                     ):
    
    # Load, chunk and index the contents of the blog.
    loader_url =SeleniumURLLoader( list_urls)
    docs = loader_url.load()
    
    splits = split_documets(docs, 
                      spliter=spliter, 
                      tokens=tokens, 
                      overlap=overlap,
                      embeedding_model=embeedding_model,
                      )
                     
    
    return splits, docs
    
def load_pdf_documets(list_pdfs, 
                      spliter="nltk", 
                      tokens=500, 
                      overlap=0.1,
                      embeedding_model=OpenAIEmbeddings(),
                     ):
    
    
    # Load, chunk and index the contents of the blog.
    pages = []
    
    for f in list_pdfs:
        try:
            loader = PyPDFLoader(f,)
            pages += loader.load_and_split()
        except Exception as e:
            print(e)
    
    
    splits = split_documets(pages, 
                      spliter=spliter, 
                      tokens=tokens, 
                      overlap=overlap,
                      embeedding_model=embeedding_model, 
                      )
                      
    return splits, pages

    
    
def question2chunk(
                chunk, 
                llm=ChatOpenAI(model="gpt-4o-mini",temperature=0.0)
                ):
    """
    Use the language model to create a question for a chunks
    """
    
    SYSTEM_PROMPT = """
    You are an AI assistant and an expert of the quantum physics. 
    """
    # Create a question for the chunk
    if hasattr(chunk, 'page_content'):
        text = chunk.page_content
    else:
        text = chunk
        
    USER_PROMPT = f"""
    Carefully read the part of the text enclosed in the ###context tag
    and write the question for that part that this text answers.
    ###context
    {text}

    """
    return llm.invoke([
           {"role": "system", "content": SYSTEM_PROMPT},
           {"role": "user", "content": USER_PROMPT},
           ]).content


def get_questioins(path):
    """
    Read a file with questions
    """
    loader = TextLoader(path)
    docs = loader.load()
    texts = docs[0].page_content.split('\n')
    questions = []
    for q in  texts:
        if "?" in q:
            questions.append(q)
    return questions

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    

def get_ideal_answer(question, context, llm):
    
    '''
    Get an ideal answer summarizing the context  for  a given question
    '''

    # Define prompt
    SYSTEM_PROMPT = """
    Human: You are an AI assistant , an expert in machine learning and the quantum physics. 
    You are able to find answers to the questions from the contextual passage snippets provided.
    \n\n
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags 
    to provide an answer to the question enclosed in <question> tags.
    <context> : {context}
    
    <question> :{question}
    
    if you do not know the answer, just say that you don't know. Do not make up an answer.
    """

    prompt = PromptTemplate.from_template(SYSTEM_PROMPT + USER_PROMPT)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    if type(context) == list:
        context = " ".join(context)
        

    return llm_chain.invoke({"context": context,
                  "question": question
                 })['text']
                 
                 
def get_embs(
    chunks, # list of  chunks
    embedding, # embbeding model of OpenAi
    question=True, # if True, use the question2chunk function  to generate a question for a chunk
    verbose=False # if True, Show generated qeustions for chuncks
    ):
    """
    Get embeddings for a list of chunks
    if question is True gives a list of queations embeddings else  a list of chunks embeddings
    """
    # make a list of striing with chunks or questions for chuncks 
    q_list = [question2chunk(t) for t in chunks] if question else [t.page_content for t in chunks]
    
    #embedding = OpenAIEmbeddings(model =embedding, show_progress_bar =  verbose)
    emb = embedding.embed_documents(q_list,)
    
    # Show generated qeustions for chuncks
    if  verbose:
        n = 1
        for  q, c in zip(q_list, chunks):
            print("#N", n)
            print(q)
            print()
            print(c.page_content)
            print("*"*10)
            n +=1
            
    return emb,  q_list
    
def get_context(
                question, 
                vecstore, 
                collection_name, 
                embedding, 
                top_k=10,
                verbose=False,
               ):
    #print(embedding(q).shape)
    
    search_res = vecstore.search(
         collection_name=collection_name,
         data=[
              embedding(question)
              ],  # Use the `emb_texts` function to convert the question to an embedding vector
         limit=top_k,  # Return top k results
         search_params={"metric_type": "IP", "params": {}},  # Inner product distance
         output_fields=["text"],  # Return the text field
             )

    retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
                                     ]
    if verbose:
        print("Retrieved lines with distances:")
        for line, distance in retrieved_lines_with_distances:
            print(f"- {line}")
            print(f"  Distance: {distance}")
            print()
         
    
    #context = "\n".join(
    #[line for (line, dist) in retrieved_lines_with_distances]
    #      )
    
    #context = [line for (line, dist) in retrieved_lines_with_distances]
    context = [line.replace("{", " ").replace("}", " ") for (line, dist) in retrieved_lines_with_distances]
    
    
    return context
    

def get_datasamples(questions, 
                    answers,
                    milvus_client, 
                    collection_name, 
                    embedding = lambda q: OpenAIEmbeddings().embed_query(q),
                    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.),
                    top_k=10,
                   ):
    contexts = []
    ideal_answers = []

    for i,q in enumerate(questions):
        context = get_context(q[2:], milvus_client, collection_name, embedding, top_k=top_k)
        #print(i)
        #print(q)
        #print(context)
        i_answer = get_ideal_answer(q[2:], context, llm=llm)
    
        contexts.append(context)
        ideal_answers.append(i_answer)

    data_samples = {'question': questions, 
           'ground_truth': answers,
           "contexts": contexts,
          "answer": ideal_answers
          }
    return Dataset.from_dict(data_samples)
