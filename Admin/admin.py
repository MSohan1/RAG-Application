import boto3
import streamlit as st
import os
import uuid

## s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="ap-northeast-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)


def get_unique_id():
    return str(uuid.uuid4())

## Split the pages/ Text  into  chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## Creating A vector store to store the text
import os

def create_vector_store(request_id, split_doc):
    print("[*] Creating FAISS vector store...")

    vector_store_faiss = FAISS.from_documents(split_doc, bedrock_embeddings)
    print(f"[✓] Vector store created.")

    folder_path = os.path.join("/tmp", request_id)

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Save FAISS index and pkl
    vector_store_faiss.save_local(folder_path)
    print(f"[✓] Saved FAISS to folder: {folder_path}")

    # Expected paths
    faiss_file = os.path.join(folder_path, "index.faiss")
    pkl_file = os.path.join(folder_path, "index.pkl")

    # Check file existence
    if not os.path.exists(faiss_file):
        raise FileNotFoundError(f"FAISS file not found: {faiss_file}")
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"PKL file not found: {pkl_file}")

    # Upload to S3
    print("[*] Uploading to S3...")
    s3_client.upload_file(Filename=faiss_file, Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=pkl_file, Bucket=BUCKET_NAME, Key="my_faiss.pkl")
    print("[✓] Upload complete.")

    return True



def main():
    st.write("This is Admin Site for Chat with PDF demo")
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request Id: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        #Split Pages
        split_doc = split_text(pages,1000,200)
        st.write(f"Total Pages after splitting : {len(pages)}")

        st.write("=====================")
        st.write(split_doc[0])
        st.write("=====================")
        st.write(split_doc[1])


        st.write("Creating a vector Store")
        result = create_vector_store(request_id,split_doc)

        if result:
            st.write("The Data is inserted into Vector Store")
        else:
            st.write("Faild to insert into Vector Store")





if __name__ == "__main__":
    main()