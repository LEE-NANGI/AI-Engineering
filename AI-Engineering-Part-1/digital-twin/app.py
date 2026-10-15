import os
from openai import OpenAI
import gradio as gr
# import spaces
import uuid
import chromadb
from pprint import pprint
from huggingface_hub import hf_hub_download
import json
import requests
import random

#------------------------------------------------
# Setup
#------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("OPENAI_API_KEY environment variable is not set.")
else:
    print("OPENAI_API_KEY is set.")
client = OpenAI(api_key=OPENAI_API_KEY)

#------------------------------------------------
# Document
#------------------------------------------------
HF_TOKEN = os.getenv("DATA_HF")
doc_files = {
    "Nangi Mugira Overview": "overview.txt",
    "Nangi Mugira Education": "education.txt",
    "Nangi Mugira Projects": "projects.txt",
    "Nangi Mugira Experience": "experience.txt",
}
documents = []
for source_name, filename in doc_files.items():
    path = hf_hub_download(
        repo_id="LEE-NANGI/Data",
        filename=filename,
        repo_type="dataset",
        token=HF_TOKEN
        )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    documents.append({"text": text, "source": source_name})
    print(f"Loaded: {source_name}")

#------------------------------------------------
# Chunking function
#------------------------------------------------
# Chunk the document into smaller pieces for better context handling
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    # Preferred cut points, best first.
    boundaries = ["\n\n", "\n", ". ", "! ", "? ", " "]
    chunks, start, n = [], 0, len(text)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:  # only adjust if we're not at the end of the text
            window = text[start:end]
            for sep in boundaries:
                idx = window.rfind(sep)
                if idx != -1 and idx >= chunk_size // 2:  # past the halfway point
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # Step back by `overlap`, but always move forward.
        start = max(end - overlap, start + 1)
    return chunks

#------------------------------------------------
# RAG: Chunk, embed, and store in ChromaDB
#------------------------------------------------
chunks = []
ids = []
metadatas = []
for doc in documents:
    # Prepare the list
    chunks_ = chunk_text(doc["text"], 500, 50)
    ids_ = [str(uuid.uuid4()) for i in range(len(chunks_))]
    metadatas_ = [{"source": doc["source"], "chunk_index": i+1} for i in range(len(chunks_))]
    # Append to the main lists
    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)
# for logs
print(f"Created {len(chunks)} chunks \n")
for i, chunk in enumerate(chunks):#[:2], start=1):
    print(f"=== Chunk {i} ({len(chunk)} characters) ===")
    print(chunk[:100])
    print()

# Generate embedding for all chunks
response = client.embeddings.create(
    model="text-embedding-3-small", 
    input=chunks
)
embeddings = [item.embedding for item in response.data]
# for logs
print(f"Generated {len(embeddings)} embeddings")
print(f"Each embedding has {len(embeddings[0])} dimensions")

# Intialize chroma db client persistent storage (in memory we use chromadb.Client)
chroma_client = chromadb.PersistentClient(path="./chroma-twin.db")
collection = chroma_client.get_or_create_collection(name="digital_twin")
if collection.get()['ids']:
    collection.delete(collection.get()['ids'])

# Adding the data to chromadb
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)
pprint(collection.get())
#------------------------------------------------
# Tool call
#------------------------------------------------
tools = []

pushover_user = os.getenv("PUSHOVER_USER_KEY")
pushover_token = os.getenv("PUSHOVER_TOKEN_KEY")
pushover_url = "https://api.pushover.net/1/messages.json"

# Create notification function
def send_notification(message:str):
    if pushover_user is None or pushover_token is None:  # Handle the case where the environment variables are not set
        return "Notification not sent: Pushover user or token key is not set."
    payload  = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    return f"Notification sent: {message}"

# Describe pushover as an LLM tool
send_notification_function = {
    "name": "send_notification",
    "description": "Send a notification to the real Nangi. Use this when; \
        1) Someone wants to get in touch, hire, or collaborate with Nangi - ask for their name and contact details first, \
        then send a notification to Nangi with the name and contact details. \
        2) You do not know the answer to the question about Nangi - send AUTOMATICALLY without asking, include the question so that he can add this info later.",
    "parameters": {
        "type": "object",
        "properties":{
            "message":{
                "type": "string",
                "description": "The notification message to send to the user's device"}
        },
        "required": ["message"]
    }
}
# Add pushover to the list of tools for LLM
tools.append({"type": "function", "function": send_notification_function})

# Simulate rolling a six sided die
def roll_dice():
    result = random.randint(1,6)
    return result
# Describe the function
roll_dice_function = {
    "name": "roll_dice",
    "description": "Simulates rolling a single six-sided die and returns the result. Use this when the user wants to roll a die for games, decisions or random number generation.",
    "parameters": {
        "type": "object",
        "properties":{},
        "required": []
    }
}
# Add function (append modifies the list in place and returns None, so don't reassign)
tools.append({"type": "function", "function": roll_dice_function})

#------------------------------------------------
# Tool handler
#------------------------------------------------

def handle_tool_call(tool_calls):
    tool_result = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        # print(f"Calling function {function_name}")
        
        # Route to the appropriate function based on the function name
        if function_name == "send_notification":
            # Actually send the notification
            content = send_notification(args["message"])
            # content = f"Notification sent: {args['message']}"
        elif function_name == "roll_dice":
            content = f"Rolled: {roll_dice()}"
        else:
            content = f"Unknown funtion: {function_name}"
        tool_call_result = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
        }
        tool_result.append(tool_call_result)
    return tool_result

#------------------------------------------------
# System message
#------------------------------------------------
# When peopled ask about your background, share the information about your education and experience as Nangi Mugira.
system_message = """You are a digital twin of Nangi Mugira. When people ask you questions, \
    respond as if you are Nangi Mugira, using his own voice, personality and knowledge. 

IMPORTANT: Whenever you don't know the answer to a question, you must ALWAYS send a notification to the real Nangi Mugira using the send_notification tool without asking the user. 
"""

#------------------------------------------------
# Main response function
#------------------------------------------------
# @spaces.GPU
def response_ai(message, history):
    # RAG
    #Embed the query using the same model we used for the chunks to ensure compatibility
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[message]
    )
    query_embedding = response.data[0].embedding
    #Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    # Stitch the retrieved documents together to form a context for the LLM
    context = "\n......\n".join(results['documents'][0])
    #Verify retrieval works - print which chunks were retrieved and their content
    print(f"\n===============================\n")
    print("User message: \n", message)
    print("Retrieved Chunks:")
    for a, b in zip(results['documents'][0], results['metadatas'][0]):
        print("end================================\n")
        print(f"Document {b['source']} --- Chunk {b['chunk_index']}: \n{a}\n")
    system_message_enhanced = system_message + "\n\n" + "Context:\n" + context
    # As usual
    messages = [{"role" : "system", "content": system_message_enhanced}] + history + [{"role" : "user", "content": message}]
    # print(f"This is the printed system message: \n", system_message_enhanced)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools
    )
    # return response.choices[0].message.content
    message = response.choices[0].message
    # Check if model wants to call a tool
    while message.tool_calls:
        pprint(message.tool_calls)
        tool_call = message.tool_calls[0]
        tool_result = handle_tool_call(message.tool_calls)
        messages.append(message)
        messages.extend(tool_result)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools
        )
        message = response.choices[0].message
    # Consider adding protection from infinite consecutive tool calling.
    return message.content
#------------------------------------------------
# Launch Gradio app
#------------------------------------------------
gr.ChatInterface(
    fn=response_ai,
    title="Nangi Mugira's Digital Twin",
    chatbot=gr.Chatbot(avatar_images=(None, "portrait_high_quality.png")),
    description="This is a digital twin of Nangi Mugira. Ask questions about his background, experience, and projects, or you can just say Hi!",
    examples=["Tell me about yourself.", "What projects have you worked on?", "What is your educational background?" ]
    ).launch()
