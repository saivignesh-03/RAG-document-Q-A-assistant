from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

app = FastAPI()

# --- Load and prepare everything ONCE, when the server starts ---
print("Loading document and building pipeline...")

loader = PyPDFLoader("data/HBO.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(pages)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="chroma_db")

tokenized_chunks = [chunk.page_content.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
llm = ChatOllama(model="llama3.1", temperature=0)

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context provided below. 
If the context does not contain enough information to answer the question, say "I don't have enough information in the document to answer that" — do not use any outside knowledge.

Context:
{context}

Question: {question}

Answer:
""")

print("Pipeline ready!")

# --- Your existing functions, unchanged ---

def keyword_search(query, k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    scored_indices = [(i, s) for i, s in enumerate(scores) if s > 0]
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, s in scored_indices[:k]]
    return [chunks[i] for i in top_indices]

def hybrid_search(query, k=3):
    semantic_results = vectorstore.similarity_search(query, k=10)
    keyword_results = keyword_search(query, k=10)
    semantic_ranks = {doc.page_content: rank for rank, doc in enumerate(semantic_results)}
    keyword_ranks = {doc.page_content: rank for rank, doc in enumerate(keyword_results)}
    all_chunks = set(semantic_ranks.keys()) | set(keyword_ranks.keys())
    rrf_scores = {}
    for chunk_text in all_chunks:
        score = 0
        if chunk_text in semantic_ranks:
            score += 1 / (60 + semantic_ranks[chunk_text])
        if chunk_text in keyword_ranks:
            score += 1 / (60 + keyword_ranks[chunk_text])
        rrf_scores[chunk_text] = score
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks[:k]

def rerank(query, candidates, top_n=3):
    pairs = [[query, chunk_text] for chunk_text, _ in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [(chunk_text, score) for (chunk_text, _), score in reranked[:top_n]]

def answer_question(question, k_hybrid=5, k_final=3):
    hybrid_candidates = hybrid_search(question, k=k_hybrid)
    reranked = rerank(question, hybrid_candidates, top_n=k_final)
    context = "\n\n---\n\n".join([text for text, score in reranked])
    prompt = rag_prompt.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response.content, reranked

# --- The actual API endpoint ---

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    answer, sources = answer_question(q.question)
    return {
        "answer": answer,
        "sources": [{"text": text[:200], "score": float(score)} for text, score in sources]
    }