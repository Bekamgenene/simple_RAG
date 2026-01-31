"""
Simple RAG System - Streamlit Web Interface
TF-IDF Vectorization & Cosine Similarity
"""

import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'document_names' not in st.session_state:
        st.session_state.document_names = []
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None
    if 'document_vectors' not in st.session_state:
        st.session_state.document_vectors = None
    if 'vectorized' not in st.session_state:
        st.session_state.vectorized = False


def vectorize_documents():
    """Convert documents to TF-IDF vectors"""
    if len(st.session_state.documents) > 0:
        st.session_state.vectorizer = TfidfVectorizer()
        st.session_state.document_vectors = st.session_state.vectorizer.fit_transform(
            st.session_state.documents
        )
        st.session_state.vectorized = True
        return True
    return False


def search_documents(query):
    """Search documents using cosine similarity"""
    if not st.session_state.vectorized or not query.strip():
        return None
    
    # Convert query to vector
    query_vector = st.session_state.vectorizer.transform([query])
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_vector, st.session_state.document_vectors)[0]
    
    # Find most relevant document
    most_relevant_idx = np.argmax(similarities)
    max_similarity = similarities[most_relevant_idx]
    
    return most_relevant_idx, max_similarity, similarities


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Simple RAG System",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🔍 Simple RAG System")
    st.markdown("**TF-IDF Vectorization & Cosine Similarity**")
    st.markdown("---")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['txt', 'text'],
            accept_multiple_files=True,
            help="Upload one or more text documents"
        )
        
        if uploaded_files:
            if st.button("Load Documents", type="primary"):
                st.session_state.documents = []
                st.session_state.document_names = []
                st.session_state.vectorized = False
                
                for uploaded_file in uploaded_files:
                    content = uploaded_file.read().decode('utf-8')
                    if content.strip():
                        st.session_state.documents.append(content)
                        st.session_state.document_names.append(uploaded_file.name)
                
                if vectorize_documents():
                    st.success(f"✓ Loaded and vectorized {len(st.session_state.documents)} document(s)!")
                else:
                    st.error("❌ Failed to vectorize documents")
        
        # Display loaded documents
        if st.session_state.documents:
            st.markdown("---")
            st.subheader("Loaded Documents")
            for i, name in enumerate(st.session_state.document_names, 1):
                st.text(f"{i}. {name}")
            
            if st.session_state.vectorized:
                st.success(f"✓ Vectorized ({st.session_state.document_vectors.shape[0]} docs)")
        
        # Option to load sample documents
        st.markdown("---")
        if st.button("Load Sample Documents"):
            st.session_state.documents = []
            st.session_state.document_names = []
            st.session_state.vectorized = False
            
            sample_dir = "sample_documents"
            if os.path.exists(sample_dir):
                for filename in os.listdir(sample_dir):
                    if filename.endswith('.txt'):
                        filepath = os.path.join(sample_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content.strip():
                                    st.session_state.documents.append(content)
                                    st.session_state.document_names.append(filename)
                        except Exception as e:
                            st.error(f"Error loading {filename}: {e}")
                
                if vectorize_documents():
                    st.success(f"✓ Loaded {len(st.session_state.documents)} sample document(s)!")
                    st.rerun()
            else:
                st.error("Sample documents folder not found")
    
    # Main content area
    if not st.session_state.vectorized:
        st.info("👈 Please upload documents or load sample documents from the sidebar to get started")
        
        # Show instructions
        with st.expander("ℹ️ How to Use", expanded=True):
            st.markdown("""
            ### Steps to Use the Simple RAG System:
            
            1. **Upload Documents**: Click on the file uploader in the sidebar and select one or more text files
            2. **Load Documents**: Click the "Load Documents" button to process and vectorize them
            3. **Enter Query**: Type your search query in the text box below
            4. **View Results**: See the most relevant document and similarity scores
            
            ### Example Queries:
            - "I want to find information on Apples"
            - "Tell me about oranges"
            - "What do you know about technology?"
            
            ### How It Works:
            - Documents are converted to **TF-IDF vectors** (Term Frequency-Inverse Document Frequency)
            - Your query is also converted to a vector using the same model
            - **Cosine similarity** is calculated between your query and all documents
            - The document with the highest similarity score is the most relevant
            """)
    
    else:
        # Query section
        st.header("🔎 Search Query")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your query:",
                placeholder="Example: I want to find information on Apples",
                help="Type your search query here"
            )
        
        with col2:
            search_button = st.button("Search", type="primary", use_container_width=True)
        
        # Perform search
        if (query and search_button) or (query and st.session_state.get('last_query') != query):
            st.session_state.last_query = query
            
            result = search_documents(query)
            
            if result:
                most_relevant_idx, max_similarity, similarities = result
                
                # Display results
                st.markdown("---")
                st.header("📊 Results")
                
                # Most relevant document
                st.success(f"**🎯 Most Relevant Document:** {st.session_state.document_names[most_relevant_idx]}")
                st.metric("Similarity Score", f"{max_similarity:.4f}")
                
                # All documents ranked
                st.subheader("📋 All Documents Ranked by Relevance")
                
                ranked_indices = np.argsort(similarities)[::-1]
                
                # Create a table
                results_data = []
                for rank, idx in enumerate(ranked_indices, 1):
                    results_data.append({
                        "Rank": rank,
                        "Document": st.session_state.document_names[idx],
                        "Similarity Score": f"{similarities[idx]:.4f}",
                        "Progress": similarities[idx]
                    })
                
                for item in results_data:
                    col1, col2, col3 = st.columns([1, 4, 2])
                    with col1:
                        st.write(f"**#{item['Rank']}**")
                    with col2:
                        st.write(item['Document'])
                    with col3:
                        st.progress(float(item['Progress']))
                        st.caption(item['Similarity Score'])
                
                # Document preview
                st.markdown("---")
                with st.expander("📄 View Document Preview", expanded=True):
                    st.subheader(st.session_state.document_names[most_relevant_idx])
                    preview_length = st.slider("Preview length (characters)", 100, 1000, 500, 50)
                    preview = st.session_state.documents[most_relevant_idx][:preview_length]
                    st.text_area(
                        "Content:",
                        preview + ("..." if len(st.session_state.documents[most_relevant_idx]) > preview_length else ""),
                        height=300
                    )
        
        elif query and not search_button:
            st.info("Click 'Search' to find relevant documents")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <small>Simple RAG System | TF-IDF + Cosine Similarity | No External LLM</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
