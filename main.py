import streamlit as st
import os

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import pickle
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import base64
import tempfile
# Import LangChain components
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Page configuration
st.set_page_config(
    page_title="PDF Chat with Gemini",
    page_icon="📚",
    layout="wide",
)

# Function to display PDF in the sidebar
def display_pdf(pdf_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # Write PDF content to the temp file
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    # Read the file
    with open(tmp_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    # Clean up the temp file
    os.unlink(tmp_path)
    
    # PDF display
    pdf_display = f"""
    <div style="display: flex; justify-content: center; width: 100%;">
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="400px" 
            type="application/pdf"
            style="border: 1px solid #ddd; border-radius: 5px;"
        >
        </iframe>
    </div>
    """
    
    # Add a download button as backup
    st.download_button(
        label="📥 Download PDF",
        data=pdf_file,
        file_name=pdf_file.name,
        mime="application/pdf",
    )
    
    return pdf_display

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to get Google embeddings
def get_embeddings(text):
    try:
        if not GOOGLE_API_KEY:
            st.error("Please provide a Google API Key for embeddings.")
            return None
            
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector = embeddings.embed_query(text)
        return vector
    except Exception as e:
        st.error(f"Error generating Google embeddings: {e}")
        return None

# Function to create vector store
def get_vector_store(text_chunks):
    st.info("Building vector store...")
    embeddings_list = []
    total_chunks = len(text_chunks)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Process each chunk
    for i, chunk in enumerate(text_chunks):
        embedding = get_embeddings(chunk)
        if embedding:
            embeddings_list.append(embedding)
        
        # Update progress
        progress_bar.progress((i + 1) / total_chunks)
    
    if not embeddings_list:
        st.error("Failed to generate embeddings for text chunks")
        return False
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list).astype('float32')
    
    # Create FAISS index
    dimension = len(embeddings_list[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # Save index and text chunks (for local use)
    faiss.write_index(index, "faiss.index")
    with open("text_chunks.pkl", "wb") as f:
        pickle.dump(text_chunks, f)
    
    # Also save to session state (for cloud use)
    with open("faiss.index", "rb") as f:
        st.session_state.faiss_index = f.read()
    st.session_state.text_chunks = text_chunks
    
    st.success(f"Vector store created with {len(embeddings_list)} embeddings of dimension {dimension}")
    return True

# Function to create the conversational chain with modern LangChain syntax
def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant powered by Gemini 2.0 Flash. You're helping answer questions about documents.
    
    Context information is below:
    {context}
    
    Given the context information and not prior knowledge, answer the following question:
    {question}
    
    If the answer cannot be found in the context, politely state that you don't have enough information to answer accurately.
    Provide clear, concise responses and cite specific parts of the document when appropriate.
    """

    # Using gemini-2.0-flash for better response quality and speed
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.9)

    prompt = PromptTemplate.from_template(prompt_template)
    
    # Create the modern stuff documents chain
    chain = create_stuff_documents_chain(model, prompt)
    
    return chain

# Function to process user questions
def process_user_question(user_question):
    try:
        # Get embedding for question
        question_embedding = get_embeddings(user_question)
        if question_embedding is None:
            return "Failed to generate embeddings for your question."
        
        # Try to load from session state first (for cloud compatibility)
        if 'faiss_index' in st.session_state and 'text_chunks' in st.session_state:
            # Create a temporary file with the saved index bytes
            with open("temp_faiss.index", "wb") as f:
                f.write(st.session_state.faiss_index)
                
            # Load FAISS index
            index = faiss.read_index("temp_faiss.index")
            text_chunks = st.session_state.text_chunks
        else:
            # Fall back to file loading (local mode)
            index = faiss.read_index("faiss.index")
            with open("text_chunks.pkl", "rb") as f:
                text_chunks = pickle.load(f)
        
        # Search for similar chunks
        question_embedding_array = np.array([question_embedding]).astype('float32')
        D, I = index.search(question_embedding_array, k=5)
        
        # Get the most relevant chunks
        relevant_chunks = [text_chunks[idx] for idx in I[0]]
        
        # Convert chunks to Document objects for the new API
        documents = [Document(page_content=chunk) for chunk in relevant_chunks]
        
        # Get conversational chain
        chain = get_conversational_chain()
        
        # Use the chain's invoke method to get a response
        response = chain.invoke({
            "context": documents, 
            "question": user_question
        })
        
        return response
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Main application
def main():
    global GOOGLE_API_KEY
    
    with st.sidebar:
        GOOGLE_API_KEY = st.text_input("Enter your Google API Key:", type="password")
        if GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
            genai.configure(api_key=GOOGLE_API_KEY)
    
    # Display custom title with local Gemma logo
    try:
        image_path = "gemini_logo.png"  # Path for local development
        
        # Convert the local image to base64
        with open(image_path, "rb") as img_file:
            local_img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Create a unified title with perfect alignment
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <h2 style="margin: 0; margin-right: 20px;">PDF Chat with</h2>
                <img src="data:image/jpeg;base64,{local_img_base64}" height="80" style="margin-top: 5px;">
            </div>
            """, 
            unsafe_allow_html=True
        )
    except Exception as e:
        # Fallback to embedded logo if local file not found (for cloud deployment)
        st.title("PDF Chat with Gemini")
    
    st.caption("Upload your PDFs and chat with your documents using Gemini 2.0 Flash")
    
    # Check if Google API is configured
    if GOOGLE_API_KEY:
        st.success("Successfully connected to Google Generative AI")
        
        # Sidebar for file uploads and processing
        with st.sidebar:
            st.header("Document Processing")
            pdf_docs = st.file_uploader("Upload your PDF documents", accept_multiple_files=True, type="pdf")
            
            # Store PDFs in session state for later use
            if pdf_docs:
                st.session_state.pdf_docs = pdf_docs
                
                # Add a selector for PDF preview if multiple PDFs are uploaded
                if len(pdf_docs) > 1:
                    pdf_names = [pdf.name for pdf in pdf_docs]
                    selected_pdf = st.selectbox("Select a PDF to preview", pdf_names)
                    selected_pdf_file = [pdf for pdf in pdf_docs if pdf.name == selected_pdf][0]
                    
                    # Display the selected PDF
                    st.header("PDF Preview")
                    # Try another way of displaying
                    st.write("If the PDF viewer below doesn't work, use the download button")
                    pdf_html = display_pdf(selected_pdf_file)
                    st.markdown(pdf_html, unsafe_allow_html=True)
                    
                elif len(pdf_docs) == 1:
                    # If only one PDF, display it directly
                    st.header("PDF Preview")
                    st.write("If the PDF viewer below doesn't work, use the download button")
                    pdf_html = display_pdf(pdf_docs[0])
                    st.markdown(pdf_html, unsafe_allow_html=True)
                
                # Process button and document handling
                process_button = st.button("Process Documents")
                
                if process_button and pdf_docs:
                    with st.spinner("Processing your documents..."):
                        # Process the PDFs
                        raw_text = get_pdf_text(pdf_docs)
                        st.session_state.total_text_length = len(raw_text)
                        
                        # Get text chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.total_chunks = len(text_chunks)
                        
                        # Create vector store
                        success = get_vector_store(text_chunks)
                        
                        if success:
                            st.success(f"✅ Done! Processed {len(pdf_docs)} documents into {len(text_chunks)} chunks")
                            st.session_state.documents_processed = True
                        else:
                            st.error("Failed to process documents")
                
                # Display stats if documents are processed
                if 'documents_processed' in st.session_state and st.session_state.documents_processed:
                    st.subheader("Document Stats")
                    st.write(f"📄 Documents: {len(pdf_docs) if pdf_docs else 0}")
                    if 'total_text_length' in st.session_state:
                        st.write(f"📊 Total text length: {st.session_state.total_text_length} characters")
                    if 'total_chunks' in st.session_state:
                        st.write(f"🧩 Total chunks: {st.session_state.total_chunks}")
        
        # Main area for chat
        if 'documents_processed' not in st.session_state:
            st.info("Please upload and process documents before asking questions")
        else:
            st.success("Documents processed successfully!")
                
        # Chat interface
        if 'messages' not in st.session_state:
            st.session_state.messages = []
                
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                    
        # Chat input
        if 'documents_processed' in st.session_state and st.session_state.documents_processed:
            user_question = st.chat_input("Ask a question about your documents...")
                
            if user_question:
                # Add user question to chat history
                st.session_state.messages.append({"role": "user", "content": user_question})
                    
                # Display user question
                with st.chat_message("user"):
                    st.markdown(user_question)
                        
                # Generate and display response
                
                with st.spinner("Thinking..."):
                    with st.chat_message("assistant"):
                        response = process_user_question(user_question)
                    st.markdown(response)
                            
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please enter your Google API Key in the sidebar to proceed")

if __name__ == "__main__":
    main()