from multiprocessing import context
import os
from dotenv import load_dotenv
import chainlit as cl

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

    #taking the ss image of the page
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img_path = f"temp_pg{page_num}.png"
    pix.save(img_path)

    #taking the caption of the page
    text = page.get_text()
    caption = ""
    for line in text.split("\n"):                   #splitting the text into lines
        if any(x in line for x in ["(Illustration", "(Photo", "(Reproduced"]):
            caption = line.strip()
            break
    doc.close()
    return img_path, caption
    
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
        model="inclusionai/ring-2.6-1t:free",
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

    question_answer_chain = create_stuff_documents_chain(
        chatModel,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    cl.user_session.set("rag_chain", rag_chain)

    await cl.Message(
        content="Hi! I am your Medical Assistant 👨‍⚕️\nAsk me anything."
    ).send()


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

    response = await cl.make_async(rag_chain.invoke)({"input": message.content})

    answer = response["answer"]
    context = response["context"] # the three chunks from pinecone

    # get unique page numbers
    pages = []
    for doc in context:
        page = doc.metadata.get("page", None)
        if page is not None and page not in pages:
            pages.append(page)

    # build sources text
    pages_text = ", ".join([f"Page {p + 1}" for p in pages])
    final_answer = f"{answer}\n\n---\n**Source:** Gale Encyclopedia of Medicine, 2nd Edition Vol.1\n📖 {pages_text}"

    # extract images from retrieved pages
    elements = []
    for page_num in pages:
        img_path, caption = get_page_image_and_caption(int(page_num))
        if img_path:
            elements.append(
                cl.Image(path=img_path, name=f"page_{page_num}", display="inline")
            )
            if caption:
                final_answer += f"\n\n🖼️ *{caption}*"

    # send answer with images and feedback buttons
    actions = [
        cl.Action(name="thumbs_up", payload={"answer": final_answer}, label="👍"),
        cl.Action(name="thumbs_down", payload={"answer": final_answer}, label="👎")
    ]

    await cl.Message(content=final_answer, elements=elements, actions=actions).send()