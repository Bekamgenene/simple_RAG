"""
Simple RAG System using TF-IDF and Cosine Similarity
This application finds the most relevant document based on user query
without using any external LLM.
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRAG:
    def __init__(self):
        self.documents = []
        self.document_names = []
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None
    
    def load_documents(self):
        """Ask user to upload/specify multiple documents"""
        print("\n" + "="*60)
        print("DOCUMENT UPLOAD")
        print("="*60)
        print("Enter document file paths (one per line).")
        print("Type 'done' when finished.\n")
        
        while True:
            file_path = input("Enter document path (or 'done'): ").strip()
            
            if file_path.lower() == 'done':
                if len(self.documents) == 0:
                    print("⚠️  No documents loaded. Please add at least one document.")
                    continue
                break
            
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        self.documents.append(content)
                        self.document_names.append(os.path.basename(file_path))
                        print(f"✓ Loaded: {os.path.basename(file_path)}")
                    else:
                        print(f"⚠️  File is empty: {file_path}")
            except Exception as e:
                print(f"❌ Error reading file: {e}")
        
        print(f"\n✓ Successfully loaded {len(self.documents)} document(s).\n")
    
    def vectorize_documents(self):
        """Convert documents to TF-IDF vectors"""
        print("Converting documents to TF-IDF vectors...")
        self.document_vectors = self.vectorizer.fit_transform(self.documents)
        print(f"✓ Vectorization complete. Vector shape: {self.document_vectors.shape}\n")
    
    def query_documents(self):
        """Ask user for query and find most relevant document"""
        print("="*60)
        print("QUERY SEARCH")
        print("="*60)
        print("Enter your query to find the most relevant document.")
        print("Example: 'I want to find information on Apples'\n")
        
        query = input("Your query: ").strip()
        
        if not query:
            print("❌ Query cannot be empty.")
            return None
        
        # Convert query to vector using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # Find the most relevant document
        most_relevant_idx = np.argmax(similarities)
        max_similarity = similarities[most_relevant_idx]
        
        return most_relevant_idx, max_similarity, similarities
    
    def display_results(self, most_relevant_idx, max_similarity, similarities):
        """Display the search results"""
        print("\n" + "="*60)
        print("SEARCH RESULTS")
        print("="*60)
        
        print(f"\n🎯 Most Relevant Document: {self.document_names[most_relevant_idx]}")
        print(f"📊 Similarity Score: {max_similarity:.4f}")
        
        print("\n📋 All Documents Ranked by Relevance:")
        print("-" * 60)
        
        # Sort documents by similarity
        ranked_indices = np.argsort(similarities)[::-1]
        
        for rank, idx in enumerate(ranked_indices, 1):
            print(f"{rank}. {self.document_names[idx]:30s} (Score: {similarities[idx]:.4f})")
        
        print("\n" + "="*60)
        
        # Show a preview of the most relevant document
        show_preview = input("\nWould you like to see a preview of the most relevant document? (y/n): ").strip().lower()
        if show_preview == 'y':
            print("\n" + "-"*60)
            print(f"Preview of: {self.document_names[most_relevant_idx]}")
            print("-"*60)
            preview = self.documents[most_relevant_idx][:500]
            print(preview + ("..." if len(self.documents[most_relevant_idx]) > 500 else ""))
            print("-"*60)
    
    def run(self):
        """Main execution loop"""
        print("\n" + "="*60)
        print("SIMPLE RAG SYSTEM")
        print("TF-IDF Vectorization & Cosine Similarity")
        print("="*60)
        
        # Step 1: Load documents
        self.load_documents()
        
        # Step 2: Vectorize documents using TF-IDF
        self.vectorize_documents()
        
        # Step 3: Query loop
        while True:
            result = self.query_documents()
            
            if result:
                most_relevant_idx, max_similarity, similarities = result
                self.display_results(most_relevant_idx, max_similarity, similarities)
            
            # Ask if user wants to make another query
            print("\n")
            another = input("Would you like to make another query? (y/n): ").strip().lower()
            if another != 'y':
                break
        
        print("\n✓ Thank you for using Simple RAG System!\n")


def main():
    """Entry point of the application"""
    rag = SimpleRAG()
    rag.run()


if __name__ == "__main__":
    main()
