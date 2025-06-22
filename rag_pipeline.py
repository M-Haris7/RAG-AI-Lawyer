from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

#Step1: Setup LLM (Use DeepSeek R1 with Groq)

llm_model=ChatGroq(model="deepseek-r1-distill-llama-70b")

#Step2: Retrieve Docs

# def retrieve_docs(query):
#     return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

#Step3: Answer Question

custom_prompt_template = """
You are an AI legal assistant. Your job is to answer the user's question **only** using the information provided in the context below.

Strictly follow these rules:
1. Use only the context to form your answer.
2. Do not include any external knowledge, assumptions, or invented facts.
3. If the context does not contain enough information, respond with: "The provided context does not contain sufficient information to answer this question."
4. Be clear, concise, and factual. Do not explain unless asked to.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

User Question: {question}

Answer:
"""


def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

# question="If a government forbids the right to assemble peacefully which articles are violated and why? only mention the articles and do not explain them, just mention the articles that are violated"
# retrieved_docs=retrieve_docs(question)
# response = answer_query(documents=retrieved_docs, model=llm_model, query=question)
# print("AI Lawyer:", response.content)
