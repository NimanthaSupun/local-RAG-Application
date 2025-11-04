"""
Local RAG System - Main Streamlit Application
A privacy-first RAG system using Ollama, Qdrant, and Streamlit.
"""
import streamlit as st
from config import check_services, get_config_summary, TOP_K
from embedding import get_embedding, generate_answer_streaming, build_rag_prompt
from vectorstore import VectorStore
from document_processor import process_document

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Local RAG System",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------
# Initialize Services
# -------------------------
# Check that Ollama and Qdrant are running
check_services()

# Initialize vector store
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = VectorStore()

vectorstore = st.session_state.vectorstore

# -------------------------
# Header
# -------------------------
st.title("💬 Local RAG System")
st.markdown("**Ollama + Qdrant + Streamlit** | 🔒 100% Local & Private")
st.markdown("---")

# -------------------------
# Sidebar - Document Upload
# -------------------------
with st.sidebar:
    st.header("📄 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to add to your knowledge base"
    )
    
    # Process uploaded files
    if uploaded_files:
        with st.spinner("Processing documents..."):
            total_chunks = 0
            errors = []
            
            for file in uploaded_files:
                try:
                    # Process document (read, chunk, embed)
                    vectors, metadata_list, chunk_count = process_document(
                        file, 
                        get_embedding
                    )
                    
                    if chunk_count > 0:
                        # Store in Qdrant
                        vectorstore.upsert_vectors(vectors, metadata_list)
                        total_chunks += chunk_count
                    else:
                        errors.append(f"⚠️ {file.name}: No text extracted")
                        
                except Exception as e:
                    errors.append(f"❌ {file.name}: {str(e)}")
            
            # Show results
            if total_chunks > 0:
                st.success(
                    f"✅ Processed {len(uploaded_files)} file(s)\n\n"
                    f"📊 Added {total_chunks} chunks to database"
                )
            
            if errors:
                for error in errors:
                    st.warning(error)
    
    st.markdown("---")
    
    # Database management
    st.subheader("🗄️ Database")
    
    # Show collection info
    info = vectorstore.get_collection_info()
    if "error" not in info:
        st.metric("Vectors Stored", info.get("vectors_count", 0))
    
    # Clear database button
    if st.button("🗑️ Clear All Data", type="secondary"):
        vectorstore.clear_collection()
        st.success("Database cleared!")
        st.rerun()
    
    st.markdown("---")
    
    # Configuration expander
    with st.expander("⚙️ Configuration"):
        config = get_config_summary()
        for key, value in config.items():
            st.text(f"{key}: {value}")

# -------------------------
# Main - Query Interface
# -------------------------
st.subheader("🔍 Ask a Question")

# Query input
query = st.text_input(
    "Enter your question:",
    placeholder="What would you like to know?",
    help="Ask questions about your uploaded documents"
)

# Search button
if st.button("Ask", type="primary", use_container_width=True) and query:
    with st.spinner("Searching knowledge base..."):
        # Get query embedding
        query_vector = get_embedding(query)
        
        # Search similar chunks
        hits = vectorstore.search(query_vector, top_k=TOP_K)
        
        if not hits:
            st.warning("⚠️ No relevant context found. Try uploading documents first.")
        else:
            # Extract context chunks
            context_chunks = [hit.payload.get("text", "") for hit in hits]
            
            # Build prompt
            prompt = build_rag_prompt(query, context_chunks)
            
            # Generate answer with streaming
            st.subheader("🧠 Answer:")
            answer_placeholder = st.empty()
            answer = ""
            
            try:
                for chunk in generate_answer_streaming(prompt):
                    answer += chunk
                    answer_placeholder.markdown(answer + "▌")  # Cursor effect
                
                # Remove cursor
                answer_placeholder.markdown(answer)
                
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
            
            # Display sources
            st.markdown("---")
            st.markdown("### 📚 Sources")
            
            for i, hit in enumerate(hits, 1):
                with st.expander(
                    f"**Source {i}** - {hit.payload.get('source_file', 'Unknown')} "
                    f"(Score: {hit.score:.3f})"
                ):
                    st.markdown(f"**Chunk {hit.payload.get('chunk_index', 0) + 1}** "
                                f"of {hit.payload.get('total_chunks', 'N/A')}")
                    st.markdown(f"**Uploaded:** {hit.payload.get('upload_timestamp', 'N/A')}")
                    st.markdown("**Content:**")
                    st.text(hit.payload.get("text", "")[:500] + "..." 
                           if len(hit.payload.get("text", "")) > 500 
                           else hit.payload.get("text", ""))

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Made with ❤️ using Ollama, Qdrant, and Streamlit | "
    "<a href='https://github.com/NimanthaSupun/local-RAG-Application'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
