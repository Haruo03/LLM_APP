#  画像データを読み込み、tesseract ocr, opencvで分析
#  画像データの文字を抽出し、その内容に対して質問可能
#  ユーザがファイルをそれぞれアップロードできるように変更(2024-08-03更新)

import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import pytesseract
from PIL import Image
import cv2

QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "my_collection"

def init_page():
    st.sidebar.title("Option")
    if "costs" not in st.session_state:
        st.session_state.costs = []

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='jpn')
    return text

def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('コレクションが作成されました！')
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )

def get_text_and_build_vector_db(images):
    all_texts = []
    for image in images:
        image_path = f"/tmp/{image.name}"
        with open(image_path, "wb") as f:
            f.write(image.getbuffer())
        data_text = extract_text_from_image(image_path)
        if data_text:
            all_texts.append(data_text)
    if all_texts:
        with st.spinner("データを読み込んでいます ..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
            split_texts = []
            for text in all_texts:
                split_texts.extend(text_splitter.split_text(text))
            build_vector_store(split_texts)

def build_vector_store(texts):
    qdrant = load_qdrant()
    qdrant.add_texts(texts)

def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        search_type="similarity",
        search_kwargs={"k":4}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa(query)
    return answer, cb.total_cost

def main():
    init_page()
    st.title("画像からテキストを抽出して質問してみよう！")

    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    uploaded_files = st.file_uploader("画像ファイルをアップロードしてください", type=["png", "jpeg", "jpg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Image: {uploaded_file.name}", use_column_width=True)

        try:
            get_text_and_build_vector_db(uploaded_files)
            st.success("ベクトルDBの作成が完了しました")

            query = st.text_input("Query: ", key="input")
            if query:
                qa = build_qa_model(llm)
                if qa:
                    with st.spinner("回答を作成しています..."):
                        answer, cost = ask(qa, query)
                    st.session_state.costs.append(cost)
                    if answer:
                        st.text_area("Answer", value=answer, height=200)
        except Exception as e:
            st.error(f"エラーが発生しました:{e}")

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == "__main__":
    main()
