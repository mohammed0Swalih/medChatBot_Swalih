from multiprocessing import context
import os
from dotenv import load_dotenv
import chainlit as cl
import whisper
import numpy as np
import base64
import httpx #converts py dict to json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import fitz

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

PDF_PATH = "data/Medical_Book.pdf"

def get_page_image_and_caption(page_num):
    doc = fitz.open(PDF_PATH)
    page = doc[int(page_num)]
    
    text = page.get_text()
    caption = ""
    for line in text.split("\n"):
        if any(x in line for x in ["(Illustration", "(Photo", "(Reproduced"]):
            caption = line.strip()
            break
    
    if not caption:
        doc.close()
        return None, None
    
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img_path = f"temp_page_{int(page_num)}.png"
    pix.save(img_path)
    
    doc.close()
    return img_path, caption


async def analyze_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    #calling openrouter with httpx (vision model)
    async with httpx.AsyncClient() as client:
        response = await client.post(    #POST API req to openrouter        
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemma-4-26b-a4b-it:free",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}" #this tells the format of the image to the model
                                }
                            },
                            {
                                "type": "text",
                                "text": (
                                    "You are a medical image analysis assistant. "
                                    "Look at this image carefully and: "
                                    "1. Identify the medical condition, symptom, or anatomical feature shown. "
                                    "2. Name the condition clearly. "
                                    "3. Give a brief 1-2 sentence description. "
                                    "Be specific and use proper medical terminology. "
                                    "Your response will be used to search a medical encyclopedia."
                                )
                            }
                        ]
                    }
                ],
                "max_tokens": 200
            },
            timeout=30.0
        )
        result = response.json() #Converts the API response from raw HTTP response into a Python dictionary
        
        # handle errors
        if "choices" not in result:
            error = result.get("error", {}).get("message", "Unknown error")
            return f"Image analysis failed: {error}"
        
        return result["choices"][0]["message"]["content"]


@cl.on_chat_start
async def start():

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medchatbot",
        embedding=embedding
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    chatModel = ChatOpenAI(
        model="arcee-ai/trinity-large-thinking:free",
        temperature=0.7,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )

    system_prompt = (
        "You are a medical assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Keep answers short and concise.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    cl.user_session.set("rag_chain", rag_chain)

    stt_model = whisper.load_model("small")
    cl.user_session.set("stt_model", stt_model)

    await cl.Message(
        content="Hi! I am your Medical Assistant 👨‍⚕️\nAsk me anything."
    ).send()


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    print(f"Chunk received! isStart: {chunk.isStart}, data length: {len(chunk.data)}")
    audio_chunks = cl.user_session.get("audio_chunks")
    audio_chunks.append(chunk.data)
    cl.user_session.set("audio_chunks", audio_chunks)


@cl.on_audio_end
async def on_audio_end():
    audio_chunks = cl.user_session.get("audio_chunks")
    audio_data = np.frombuffer(b"".join(audio_chunks), dtype=np.int16).astype(np.float32) / 32768.0

    stt_model = cl.user_session.get("stt_model")
    result = stt_model.transcribe(audio_data)
    query = result["text"].strip()

    if not query:
        await cl.Message(content="I couldn't hear anything. Please try again!").send()
        return

    await cl.Message(content=f"🎤 *You said:* {query}").send()

    rag_chain = cl.user_session.get("rag_chain")
    response = await cl.make_async(rag_chain.invoke)({"input": query})

    answer = response["answer"]
    context = response["context"]

    pages = []
    for doc in context:
        page = doc.metadata.get("page", None)
        if page is not None and int(page) not in pages:
            pages.append(int(page))

    pages_text = ", ".join([f"Page {p + 1}" for p in pages])
    final_answer = f"{answer}\n\n---\n**Source:** Gale Encyclopedia of Medicine, 2nd Edition Vol.1\n📖 {pages_text}"

    elements = []
    for page_num in pages:
        img_path, caption = get_page_image_and_caption(page_num)
        if img_path:
            elements.append(cl.Image(path=img_path, name=f"page_{page_num}", display="inline"))
            if caption:
                final_answer += f"\n\n🖼️ *{caption}*"

    actions = [
        cl.Action(name="thumbs_up", payload={"answer": final_answer}, label="👍"),
        cl.Action(name="thumbs_down", payload={"answer": final_answer}, label="👎")
    ]

    await cl.Message(content=final_answer, elements=elements, actions=actions).send()


@cl.action_callback("thumbs_up")
async def on_thumbs_up(action: cl.Action):
    await cl.Message(content="Thanks for the feedback! 👍").send()
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\nLIKED:\n{action.payload['answer']}\n")


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action: cl.Action):
    await cl.Message(content="Thanks for the feedback! We'll improve it 👌").send()
    with open("feedback_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\nDISLIKED:\n{action.payload['answer']}\n")


@cl.on_message
async def main(message: cl.Message):

    greetings = ["hi", "hello", "hey"]
    if message.content.lower() in greetings:
        await cl.Message(content="Hello 👋 Ask me any medical question.").send()
        return

    rag_chain = cl.user_session.get("rag_chain")
    if not rag_chain:
        await cl.Message(content="RAG chain not initialized.").send()
        return

    # check if user uploaded an image
    if message.elements:
        for element in message.elements: #looping through all images user uploaded
            if "image" in element.mime: #checking if the element is an image like jpeg or png
                await cl.Message(content="🔍 Analyzing your image...").send()

                #analyze_image function sends it to Gemma vision model and gets back a description.
                vision_response = await analyze_image(element.path) # element.path is the path to the image file saved by chainlit
                await cl.Message(content=f"🧠 **Vision Analysis:** {vision_response}").send()

                query = vision_response # we use the vision model's response as the query to the RAG chain

                response = await cl.make_async(rag_chain.invoke)({"input": query})
                answer = response["answer"]
                context = response["context"]

                pages = []
                for doc in context:
                    page = doc.metadata.get("page", None)
                    if page is not None and int(page) not in pages:
                        pages.append(int(page))

                pages_text = ", ".join([f"Page {p + 1}" for p in pages])
                final_answer = f"{answer}\n\n---\n**Source:** Gale Encyclopedia of Medicine, 2nd Edition Vol.1\n📖 {pages_text}"

                elements = []
                for page_num in pages:
                    img_path, caption = get_page_image_and_caption(page_num)
                    if img_path:
                        elements.append(cl.Image(path=img_path, name=f"page_{page_num}", display="inline"))
                        if caption:
                            final_answer += f"\n\n🖼️ *{caption}*"

                actions = [
                    cl.Action(name="thumbs_up", payload={"answer": final_answer}, label="👍"),
                    cl.Action(name="thumbs_down", payload={"answer": final_answer}, label="👎")
                ]

                await cl.Message(content=final_answer, elements=elements, actions=actions).send()
                return

    # normal response
    response = await cl.make_async(rag_chain.invoke)({"input": message.content})

    answer = response["answer"]
    context = response["context"]

    pages = []
    for doc in context:
        page = doc.metadata.get("page", None)
        if page is not None and page not in pages:
            pages.append(page)

    pages_text = ", ".join([f"Page {p}" for p in pages])
    final_answer = f"{answer}\n\n---\n**Source:** Gale Encyclopedia of Medicine, 2nd Edition Vol.1\n📖 {pages_text}"

    elements = []
    for page_num in pages: #loops through each page ss
        img_path, caption = get_page_image_and_caption(int(page_num))
        if img_path:
            elements.append(cl.Image(path=img_path, name=f"page_{page_num}", display="inline"))
            if caption:
                final_answer += f"\n\n🖼️ *{caption}*"

    actions = [
        cl.Action(name="thumbs_up", payload={"answer": final_answer}, label="👍"),
        cl.Action(name="thumbs_down", payload={"answer": final_answer}, label="👎")
    ]

    await cl.Message(content=final_answer, elements=elements, actions=actions).send()