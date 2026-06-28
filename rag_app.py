import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter

load_dotenv()

# -----------------------------
# STEP 1: Load PDF #check the current folder and find where the files are 
# -----------------------------
pdf_files = [f for f in os.listdir() if f.endswith(".pdf")]
if not pdf_files:
    print("❌ No PDF found!")
    exit()
#takes first file from folder
PDF_FILE = pdf_files[0]
print(f"📄 Using PDF: {PDF_FILE}")#show which is being used

converter = DocumentConverter()
result = converter.convert(PDF_FILE)#convert pdf into structured pdf
full_text = result.document.export_to_markdown() #extract text from pdf markdown

documents = [Document(page_content=full_text, metadata={"source": PDF_FILE})]
#wraps text into langchain
# -----------------------------
# STEP 2: Split Text
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# -----------------------------
# STEP 3: Embeddings + FAISS
# -----------------------------
embeddings = HuggingFaceEmbeddings(  #convert text into numbers
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# STEP 4: Load LLM
# -----------------------------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=256,
    temperature=0.3,
    device=-1
)

llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# STEP 5: Ask Function
# -----------------------------
def ask_question(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Answer the question based ONLY on the context below.
If not found, say "I cannot find this information in the PDF."

Context:
{context}

Question: {query}

Answer:"""

    return llm.invoke(prompt).strip()

# -----------------------------
# STEP 6: Chat Loop
# -----------------------------
print("\n✅ SYSTEM READY! Ask questions (type 'exit' to quit)\n")

while True:
    query = input("❓ YOUR QUESTION: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        break
    if query:
        answer = ask_question(query)
        print("\n📝 ANSWER:\n", answer if answer else "[No answer generated]")