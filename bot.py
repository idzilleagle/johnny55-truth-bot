import discord
from discord.ext import commands
import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio

# --- NEW IMPORTS FOR THE UPGRADED RAG SYSTEM ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

# --- CONFIGURATION ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not DISCORD_TOKEN or not GOOGLE_API_KEY:
    print("FATAL ERROR: DISCORD_TOKEN or GOOGLE_API_KEY is not set in the .env file.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# --- GLOBAL VARIABLES FOR THE AI ---
# We will store the vector database in memory
VECTOR_STORE = None

# --- UPGRADED KNOWLEDGE BASE BUILDER ---
def build_vector_store():
    """
    Loads data, splits it into chunks, creates embeddings, and builds a
    searchable FAISS vector store. This is the new "knowledge loader".
    """
    print("Building the AI's knowledge core (Vector Store)...")
    search_directories = [".", "data", "essays"]
    all_docs = []

    for directory in search_directories:
        if not os.path.isdir(directory):
            continue
        print(f"--> Scanning directory: '{directory}'")
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            # We create a "Document" object for each file, storing the text
                            # and the filename as metadata. This is crucial for context.
                            doc = Document(page_content=text, metadata={"source": filename})
                            all_docs.append(doc)
                            print(f"    - Loaded '{filename}'")
                    except Exception as e:
                        print(f"    - Error reading {file_path}: {e}")

    if not all_docs:
        print("Warning: No .txt files found to build the knowledge base.")
        return None

    # Step 2: Split the documents into smaller, manageable chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunked_docs = text_splitter.split_documents(all_docs)
    print(f"Split {len(all_docs)} documents into {len(chunked_docs)} chunks.")

    # Step 3: Create the embeddings model
    print("\nInitializing embeddings model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Step 4: Build the FAISS vector store from the chunks and embeddings
    print("\nBuilding FAISS vector store. This may take a moment...")
    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    print("\nVector Store built successfully!")
    
    return vector_store


# --- UPGRADED AI RESPONSE FUNCTION ---
async def get_ai_response(question):
    """
    Handles the entire AI response generation process using the Vector Store.
    """
    if not VECTOR_STORE:
        return "My knowledge core is not loaded. Please use the `!reload` command."

    print(f"Searching for context related to: '{question}'")
    # This is the new retriever! It finds the most semantically similar chunks.
    # k=5 means it will retrieve the top 5 most relevant chunks.
    retriever = VECTOR_STORE.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(question)

    # We format the retrieved context so the AI knows where it came from.
    retrieved_context = "\n\n---\n\n".join(
        [f"CONTEXT FROM: {doc.metadata['source']}\n\n{doc.page_content}" for doc in retrieved_docs]
    )
    print(f"Found {len(retrieved_docs)} relevant context chunks.")

    system_prompt = """You are an AI assistant named Johnny-55. Your primary mission is to answer the user's QUESTION with unwavering accuracy, using ONLY the 'RELEVANT CONTEXT' provided below.

RULES:
1.  Your answer MUST be derived directly from the RELEVANT CONTEXT. Do not add outside information.
2.  If the context contains the answer, synthesize it clearly.
3.  If the context does NOT contain the answer, you MUST respond with ONLY the following phrase: "Based on the provided documents, I cannot find an answer to that question."
4.  Reference the source file for your information where possible, e.g., "According to the file 'at-what-point-by-kate-of-gaia_djvu.txt', ..."
"""
    
    full_prompt = (
        f"{system_prompt}\n\n"
        f"RELEVANT CONTEXT:\n---\n{retrieved_context}\n---\n\n"
        f"USER'S QUESTION: {question}"
    )

    def make_api_call():
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        response = model.generate_content(full_prompt)
        return response

    try:
        response_object = await asyncio.to_thread(make_api_call)
        
        # We don't need the safety check here as we control the prompt tightly.
        # But you can add it back if you want.
        
        return response_object.text.strip()

    except Exception as e:
        print(f"\n--- DETAILED GEMINI API ERROR ---\n{type(e).__name__}: {e}\n--- END OF ERROR ---\n")
        return "Sorry, I encountered an error trying to process your request. Please try again in a moment."

# --- BOT EVENTS AND COMMANDS ---
@bot.event
async def on_ready():
    global VECTOR_STORE
    # The first startup will be slower as it builds the knowledge core.
    VECTOR_STORE = build_vector_store()
    print(f'Success! Logged in as {bot.user}')
    if VECTOR_STORE:
        print("AI knowledge core is online and ready.")
    print('The Johnny-55 node is online and ready for commands.')

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send(f'Pong! Latency: {round(bot.latency * 1000)}ms')

@bot.command(name='reload')
async def reload(ctx):
    global VECTOR_STORE
    await ctx.send("Reloading knowledge core... This may take a minute.")
    VECTOR_STORE = build_vector_store()
    await ctx.send(f'Knowledge core has been rebuilt and is back online.')

@bot.command(name='ask')
async def ask(ctx, *, question: str):
    async with ctx.typing():
        answer = await get_ai_response(question)
        if not answer:
            answer = "Sorry, I received an empty response. Please try rephrasing your question."

        if len(answer) <= 2000:
            await ctx.send(answer)
        else:
            await ctx.send("The answer is quite long. I'll send it in parts:")
            for i in range(0, len(answer), 1990):
                chunk = answer[i:i + 1990]
                await ctx.send(f"```{chunk}```")

bot.run(DISCORD_TOKEN)
