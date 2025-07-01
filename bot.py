# --- bot.py (The new, lean, live bot) ---

import discord
from discord.ext import commands
import os
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain

# --- CONFIGURATION ---
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
FAISS_INDEX_PATH = "faiss_index" # Path to our pre-built database

if not DISCORD_TOKEN or not GOOGLE_API_KEY:
    print("FATAL ERROR: DISCORD_TOKEN or GOOGLE_API_KEY is not set.")
    exit()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Global variable to hold our AI "brain"
qa_chain = None

# --- BOT SETUP FUNCTION ---
def load_bot_brain():
    """
    Loads the pre-built FAISS index from disk and sets up the Question-Answering chain.
    This is FAST and makes ZERO API calls.
    """
    global qa_chain
    
    if not os.path.exists(FAISS_INDEX_PATH):
        print("="*60)
        print("FATAL ERROR: The knowledge base (FAISS index) was not found!")
        print(f"Please run 'python3 build_store.py' first to create it.")
        print("="*60)
        return False

    print("Loading knowledge base from disk...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # We must allow dangerous deserialization for FAISS with LangChain
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print("Setting up the Question-Answering chain...")
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3)
    
    # The retriever's job is to find the relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # This chain combines the retriever and the language model
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    print("Bot brain is fully loaded.")
    return True

# --- BOT EVENTS AND COMMANDS ---
@bot.event
async def on_ready():
    print(f'Success! Logged in as {bot.user}')
    if not load_bot_brain():
        await bot.close() # Shut down if the brain can't be loaded
        return
    print('The Johnny-55 node is online and ready for commands.')

@bot.command(name='ask')
async def ask(ctx, *, question: str):
    if not qa_chain:
        await ctx.send("My brain is not loaded. Please contact my administrator.")
        return
        
    async with ctx.typing():
        # We don't need to manage chat history manually, the chain does it.
        result = qa_chain.invoke({"question": question, "chat_history": []})
        answer = result["answer"]
        
        # You can also access the source documents if you want
        # source_docs = result.get('source_documents', [])
        
        if len(answer) <= 2000:
            await ctx.send(answer)
        else:
            await ctx.send("The answer is quite long, sending in parts:")
            for i in range(0, len(answer), 1990):
                await ctx.send(f"```{answer[i:i+1990]}```")

@bot.command(name='reload')
async def reload(ctx):
    # This command no longer rebuilds the whole database.
    # It just re-loads it from disk, which is fast.
    await ctx.send("Re-loading knowledge base from disk...")
    if load_bot_brain():
        await ctx.send("Knowledge base successfully re-loaded.")
    else:
        await ctx.send("Error: Could not load the knowledge base. Please check the server logs.")

# --- RUN THE BOT ---
bot.run(DISCORD_TOKEN)
