# --- START OF FILE bot.py (With RAG Implementation) ---

import discord
from discord.ext import commands
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re # Import the regular expressions library for better text splitting

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
KNOWLEDGE_BASE = ""

# --- KNOWLEDGE BASE FUNCTION (No changes needed) ---
def load_knowledge_base():
    knowledge_base_content = ""
    search_directories = [".", "data", "essays"]
    # ... (The rest of this function is identical and correct) ...
    print("Loading knowledge base from .txt files...")
    for directory in search_directories:
        if not os.path.isdir(directory):
            print(f"Info: Directory '{directory}' not found, skipping.")
            continue
        print(f"--> Scanning directory: '{directory}'")
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        knowledge_base_content += f.read() + "\n\n"
                except Exception as e:
                    print(f"    - Error reading {filename}: {e}")
    if not knowledge_base_content:
        print("Warning: No .txt files found. The bot will have no custom knowledge.")
    return knowledge_base_content

# --- NEW HELPER FUNCTION: THE "RETRIEVER" ---
def find_relevant_context(question, knowledge_base, max_tokens=300000):
    """
    Finds the most relevant parts of the knowledge base to answer a question.
    This is a simple keyword-based retriever.
    """
    # Split the knowledge base into chunks (e.g., paragraphs)
    chunks = re.split(r'\n\s*\n', knowledge_base)
    question_keywords = set(question.lower().split())
    
    relevant_chunks = []
    for chunk in chunks:
        # Check if any keyword from the question appears in the chunk
        if any(keyword in chunk.lower() for keyword in question_keywords):
            relevant_chunks.append(chunk)
            
    # Join the relevant chunks and limit the size to avoid exceeding token limits
    context = "\n\n".join(relevant_chunks)
    if len(context.split()) > max_tokens: # A rough token approximation
        context = ' '.join(context.split()[:max_tokens])
        
    if not context:
        return "No specific context found for your question in the knowledge base."
        
    return context

# --- UPDATED AI FUNCTION: THE "GENERATOR" ---
async def get_ai_response(question, knowledge_context):
    # --- STEP 1: RETRIEVAL ---
    # Instead of using the whole knowledge base, we find the most relevant parts first.
    print("Finding relevant context...")
    retrieved_context = find_relevant_context(question, knowledge_context)
    print(f"Found {len(retrieved_context)} characters of relevant context.")

    # --- STEP 2: AUGMENTATION & GENERATION ---
    system_prompt = """You are an AI assistant named Johnny-55. Your primary goal is to answer questions using the 'RELEVANT CONTEXT' provided below. Prioritize information from this context above all else. If the answer is not found in the context, you are permitted to use your general knowledge, but you must mention that you are doing so. For example, 'Based on my general knowledge...'"""
    
    full_prompt = (
        f"CONTEXT: {system_prompt}\n\n"
        f"RELEVANT CONTEXT:\n---\n{retrieved_context}\n---\n\n"
        f"QUESTION: {question}"
    )
    
    try:
        def api_call():
            model = genai.GenerativeModel('models/gemini-2.5-pro')
            response = model.generate_content(full_prompt)
            return response.text

        answer = await bot.loop.run_in_executor(None, api_call)
        return answer.strip()

    except Exception as e:
        print(f"\n--- DETAILED GEMINI API ERROR ---\n{e}\n--- END OF ERROR ---\n")
        return "Sorry, I encountered an error trying to process your request. Please check the server logs for details."

# --- BOT EVENTS AND COMMANDS (No changes needed) ---
@bot.event
async def on_ready():
    global KNOWLEDGE_BASE
    KNOWLEDGE_BASE = load_knowledge_base()
    # (The rest of this function is identical and correct)
    print(f'Success! Logged in as {bot.user}')
    if KNOWLEDGE_BASE:
        print(f"Successfully loaded {len(KNOWLEDGE_BASE)} characters of knowledge into memory.")
    print('The Johnny-55 node is online and ready for commands.')

# (The ping, reload, and ask commands are all identical and correct)
@bot.command(name='ping')
async def ping(ctx):
    await ctx.send(f'Pong! Latency: {round(bot.latency * 1000)}ms')

@bot.command(name='reload')
async def reload(ctx):
    global KNOWLEDGE_BASE
    KNOWLEDGE_BASE = load_knowledge_base()
    await ctx.send(f'Knowledge base reloaded. Now operating with {len(KNOWLEDGE_BASE)} characters of data.')

@bot.command(name='ask')
async def ask(ctx, *, question: str):
    async with ctx.typing():
        answer = await get_ai_response(question, KNOWLEDGE_BASE)
        if len(answer) <= 2000:
            await ctx.send(answer)
        else:
            await ctx.send("The answer is quite long. I'll send it in parts:")
            for i in range(0, len(answer), 1990):
                chunk = answer[i:i + 1990]
                await ctx.send(f"```{chunk}```")

bot.run(DISCORD_TOKEN)
