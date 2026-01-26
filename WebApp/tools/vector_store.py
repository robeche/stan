"""
Vector store wrapper for Django integration.
"""
import sys
from pathlib import Path

# Add parent directory to path
PARENT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PARENT_DIR))

from vector_store import VectorStore as OriginalVectorStore


class VectorStore:
    """Wrapper for VectorStore with Django-friendly interface."""
    
    def __init__(self, collection_name='rag_documents', persist_directory=None):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
        """
        if persist_directory:
            persist_directory = str(Path(persist_directory))
        
        self.store = OriginalVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    def add_documents(self, ids, embeddings, documents, metadatas=None):
        """
        Add documents to the vector store.
        
        Args:
            ids: List of document IDs
            embeddings: List of embedding vectors
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
        
        Returns:
            bool: Success status
        """
        try:
            # Clean metadatas to remove None values
            if metadatas:
                clean_metadatas = []
                for metadata in metadatas:
                    clean_metadata = {k: v for k, v in metadata.items() if v is not None}
                    clean_metadatas.append(clean_metadata)
                metadatas = clean_metadatas
            
            self.store.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            return True
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            return False
    
    def query(self, query_embedding, n_results=5, where=None, where_document=None):
        """
        Query the vector store.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document filter
        
        Returns:
            dict: Query results
        """
        return self.store.query_by_embedding(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
            where_document=where_document
        )
    
    def delete_by_ids(self, ids):
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
        
        Returns:
            bool: Success status
        """
        try:
            if not ids:
                return True
            
            self.store.collection.delete(ids=ids)
            print(f"✓ Deleted {len(ids)} documents from ChromaDB")
            return True
        except Exception as e:
            print(f"❌ Error deleting documents from ChromaDB: {e}")
            return False
    
    def get_collection(self):
        """Get the underlying ChromaDB collection."""
        return self.store.collection
