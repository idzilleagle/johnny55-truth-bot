import discord
from discord.ext import commands
import os
import google.generativeai as genai
from dotenv import load_dotenv
import re # Used for RAG
import asyncio # Used for running blocking code safely

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

# --- KNOWLEDGE BASE FUNCTION (REWRITTEN FOR RECURSIVE SEARCH) ---
def load_knowledge_base():
    """
    Recursively scans specified directories for .txt files and loads them
    into a single string for the RAG context.
    """
    knowledge_base_content = ""
    # Directories to search. It will search these and all their subdirectories.
    search_directories = [".", "data", "essays"] 
    
    print("Loading knowledge base from .txt files...")
    
    for directory in search_directories:
        # Check if the base directory exists
        if not os.path.isdir(directory):
            print(f"Info: Directory '{directory}' not found, skipping.")
            continue
        
        print(f"--> Recursively scanning directory: '{directory}'")
        
        # os.walk() is the key. It goes through the directory and all subdirectories.
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    # We print the full path to confirm it's finding nested files.
                    print(f"    - Loading file: {file_path}") 
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            # Added filename as metadata into the text itself for better context
                            knowledge_base_content += f"--- START OF FILE {os.path.basename(file_path)} ---\n"
                            knowledge_base_content += f.read()
                            knowledge_base_content += f"\n--- END OF FILE {os.path.basename(file_path)} ---\n\n"
                    except Exception as e:
                        print(f"    - Error reading {file_path}: {e}")
                        
    if not knowledge_base_content:
        print("Warning: No .txt files found. The bot will have no custom knowledge.")
    return knowledge_base_content

# --- HELPER FUNCTION: THE "RETRIEVER" (No changes needed) ---
def find_relevant_context(question, knowledge_base, max_tokens=300000):
    chunks = re.split(r'\n\s*\n', knowledge_base)
    question_keywords = set(question.lower().split())
    relevant_chunks = []
    for chunk in chunks:
        if any(keyword in chunk.lower() for keyword in question_keywords):
            relevant_chunks.append(chunk)
    context = "\n\n".join(relevant_chunks)
    if len(context.split()) > max_tokens:
        context = ' '.join(context.split()[:max_tokens])
    if not context:
        return "No specific context found for your question in the knowledge base."
    return context

# --- AI RESPONSE FUNCTION (No changes needed) ---
async def get_ai_response(question, knowledge_context):
    """
    Handles the entire AI response generation process, including retrieval,
    API calls with timeouts, and safety checking.
    """
    print("Finding relevant context...")
    retrieved_context = find_relevant_context(question, knowledge_context)
    print(f"Found {len(retrieved_context)} characters of relevant context.")

    system_prompt = """You are an AI assistant named Johnny-55. Your primary goal is to answer questions using the 'RELEVANT CONTEXT' provided below. Prioritize information from this context above all else. If the answer is not found in the context, you are permitted to use your general knowledge, but you must mention that you are doing so. For example, 'Based on my general knowledge...'"""
    
    full_prompt = (
        f"CONTEXT: {system_prompt}\n\n"
        f"RELEVANT CONTEXT:\n---\n{retrieved_context}\n---\n\n"
        f"QUESTION: {question}"
    )

    def make_api_call():
        """Makes the actual API call to Google Gemini."""
        model = genai.GenerativeModel('models/gemini-1.5-pro')
        
        response = model.generate_content(
            full_prompt,
            request_options={"timeout": 60}
        )
        return response

    try:
        response_object = await asyncio.to_thread(make_api_call)

        if response_object.prompt_feedback.block_reason:
            reason = response_object.prompt_feedback.block_reason.name
            print(f"Prompt was blocked by API. Reason: {reason}")
            return f"I'm sorry, I can't answer that. My safety filters were triggered (Reason: {reason})."
        
        return response_object.text.strip()

    except Exception as e:
        print(f"\n--- DETAILED GEMINI API ERROR ---\n{type(e).__name__}: {e}\n--- END OF ERROR ---\n")
        return "Sorry, I encountered an error trying to process your request. It might have been a network timeout or an issue with the API. Please try again in a moment."

# --- BOT EVENTS AND COMMANDS (No changes needed) ---
@bot.event
async def on_ready():
    global KNOWLEDGE_BASE
    KNOWLEDGE_BASE = load_knowledge_base()
    print(f'Success! Logged in as {bot.user}')
    if KNOWLEDGE_BASE:
        print(f"Successfully loaded {len(KNOWLEDGE_BASE)} characters of knowledge into memory.")
    print('The Johnny-55 node is online and ready for commands.')

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
