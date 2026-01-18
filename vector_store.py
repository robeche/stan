"""
Módulo de Base de Datos Vectorial con ChromaDB
================================================

Sistema de almacenamiento y búsqueda vectorial para embeddings de documentos.
Integrado con el sistema RAG para retrieval eficiente.

Autor: Sistema RAG
Fecha: 2026-01-02
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import chromadb
from chromadb.config import Settings
from tqdm import tqdm


class VectorStore:
    """
    Gestor de base de datos vectorial usando ChromaDB.
    
    Características:
    - Almacenamiento persistente de embeddings
    - Búsqueda por similitud (cosine, L2, IP)
    - Filtrado por metadata
    - Múltiples colecciones
    - Operaciones batch eficientes
    """
    
    def __init__(
        self,
        persist_directory: Union[str, Path] = "output_rag/chroma_db",
        collection_name: str = "documents",
        embedding_function: Optional[callable] = None,
        distance_metric: str = "cosine"
    ):
        """
        Inicializa el vector store.
        
        Args:
            persist_directory: Directorio para persistir la BD
            collection_name: Nombre de la colección
            embedding_function: Función para generar embeddings (opcional)
            distance_metric: Métrica de distancia ('cosine', 'l2', 'ip')
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.distance_metric = distance_metric
        
        # Crear directorio si no existe
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Inicializar ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Obtener o crear colección
        self.collection = self._get_or_create_collection()
        
        print(f"✓ VectorStore inicializado")
        print(f"  Directorio: {self.persist_directory}")
        print(f"  Colección: {self.collection_name}")
        print(f"  Documentos: {self.collection.count()}")
    
    def _get_or_create_collection(self):
        """Obtiene o crea la colección."""
        try:
            collection = self.client.get_collection(
                name=self.collection_name
            )
            return collection
        except:
            # Mapear métrica de distancia
            metadata_map = {
                "cosine": {"hnsw:space": "cosine"},
                "l2": {"hnsw:space": "l2"},
                "ip": {"hnsw:space": "ip"}
            }
            
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata=metadata_map.get(self.distance_metric, {"hnsw:space": "cosine"})
            )
            return collection
    
    def add_embeddings_from_directory(
        self,
        embeddings_dir: Union[str, Path],
        show_progress: bool = True
    ) -> Dict:
        """
        Carga embeddings desde un directorio de chunks JSON.
        
        Args:
            embeddings_dir: Directorio con archivos chunk_XXXX.json
            show_progress: Mostrar barra de progreso
            
        Returns:
            Diccionario con estadísticas de la carga
        """
        embeddings_dir = Path(embeddings_dir)
        
        # Encontrar archivos de chunks
        chunk_files = sorted(embeddings_dir.glob("chunk_*.json"))
        
        if not chunk_files:
            raise ValueError(f"No se encontraron chunks en {embeddings_dir}")
        
        print(f"\nCargando {len(chunk_files)} chunks en ChromaDB...")
        
        # Preparar datos para batch insert
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        iterator = chunk_files
        if show_progress:
            iterator = tqdm(chunk_files, desc="Cargando chunks")
        
        for chunk_file in iterator:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            # Preparar datos
            chunk_id = f"chunk_{chunk_data.get('chunk_id', len(ids)):04d}"
            embedding = chunk_data.get('embedding', [])
            content = chunk_data.get('content', '')
            metadata = chunk_data.get('metadata', {})
            
            # Agregar info adicional a metadata (ChromaDB no acepta None)
            metadata_clean = {}
            for k, v in metadata.items():
                if v is not None:
                    if isinstance(v, (str, int, float, bool)):
                        metadata_clean[k] = v
                    else:
                        metadata_clean[k] = str(v)
            
            metadata_clean.update({
                'chunk_id': chunk_data.get('chunk_id', len(ids)),
                'length': chunk_data.get('length', len(content)),
                'embedding_model': chunk_data.get('embedding_model', 'unknown')
            })
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(content)
            metadatas.append(metadata_clean)
        
        # Insertar en batch
        print(f"Insertando {len(ids)} documentos en ChromaDB...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        stats = {
            'chunks_loaded': len(ids),
            'collection_name': self.collection_name,
            'total_documents': self.collection.count()
        }
        
        print(f"✓ {stats['chunks_loaded']} chunks cargados exitosamente")
        print(f"  Total en colección: {stats['total_documents']}")
        
        return stats
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        Busca documentos similares usando texto.
        
        Args:
            query_text: Texto de consulta
            n_results: Número de resultados a retornar
            where: Filtros de metadata (ej: {"chunk_id": 5})
            where_document: Filtros de documento (ej: {"$contains": "turbine"})
            
        Returns:
            Diccionario con resultados de la búsqueda
        """
        if not self.embedding_function:
            raise ValueError(
                "Se requiere embedding_function para queries de texto. "
                "Usa query_by_embedding() o proporciona embedding_function al inicializar."
            )
        
        # Generar embedding para la query
        query_embedding = self.embedding_function(query_text)
        
        return self.query_by_embedding(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where,
            where_document=where_document
        )
    
    def query_by_embedding(
        self,
        query_embedding: Union[List[float], np.ndarray],
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        Busca documentos similares usando embedding directo.
        
        Args:
            query_embedding: Vector de embedding
            n_results: Número de resultados a retornar
            where: Filtros de metadata
            where_document: Filtros de documento
            
        Returns:
            Diccionario con resultados de la búsqueda
        """
        # Convertir numpy array a lista si es necesario
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Realizar búsqueda
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Reformatear resultados
        formatted_results = {
            'query_embedding_dim': len(query_embedding),
            'n_results': len(results['ids'][0]),
            'results': []
        }
        
        for i in range(len(results['ids'][0])):
            formatted_results['results'].append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Para cosine
            })
        
        return formatted_results
    
    def get_by_ids(self, ids: List[str]) -> Dict:
        """
        Obtiene documentos específicos por sus IDs.
        
        Args:
            ids: Lista de IDs de documentos
            
        Returns:
            Diccionario con los documentos
        """
        results = self.collection.get(
            ids=ids,
            include=['documents', 'metadatas', 'embeddings']
        )
        
        return {
            'ids': results['ids'],
            'documents': results['documents'],
            'metadatas': results['metadatas'],
            'embeddings': results['embeddings']
        }
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Elimina documentos por sus IDs.
        
        Args:
            ids: Lista de IDs a eliminar
        """
        self.collection.delete(ids=ids)
        print(f"✓ {len(ids)} documentos eliminados")
    
    def update_metadata(self, id: str, metadata: Dict) -> None:
        """
        Actualiza metadata de un documento.
        
        Args:
            id: ID del documento
            metadata: Nueva metadata
        """
        self.collection.update(
            ids=[id],
            metadatas=[metadata]
        )
    
    def reset_collection(self) -> None:
        """Elimina todos los documentos de la colección."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        print(f"✓ Colección '{self.collection_name}' reseteada")
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas de la colección."""
        return {
            'collection_name': self.collection_name,
            'total_documents': self.collection.count(),
            'persist_directory': str(self.persist_directory),
            'distance_metric': self.distance_metric
        }
    
    def list_collections(self) -> List[str]:
        """Lista todas las colecciones en el cliente."""
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def peek(self, limit: int = 5) -> Dict:
        """
        Muestra una muestra de documentos.
        
        Args:
            limit: Número de documentos a mostrar
            
        Returns:
            Diccionario con muestra de documentos
        """
        return self.collection.peek(limit=limit)


def main():
    """Función principal para uso desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gestión de base de datos vectorial con ChromaDB"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # Comando: load
    load_parser = subparsers.add_parser('load', help='Cargar embeddings en ChromaDB')
    load_parser.add_argument(
        'embeddings_dir',
        type=str,
        help='Directorio con embeddings (chunks JSON)'
    )
    load_parser.add_argument(
        '--persist-dir',
        type=str,
        default='output_rag/chroma_db',
        help='Directorio para persistir ChromaDB'
    )
    load_parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='Nombre de la colección'
    )
    
    # Comando: query
    query_parser = subparsers.add_parser('query', help='Buscar en ChromaDB')
    query_parser.add_argument(
        'query_embedding_file',
        type=str,
        help='Archivo JSON con embedding de consulta'
    )
    query_parser.add_argument(
        '--persist-dir',
        type=str,
        default='output_rag/chroma_db',
        help='Directorio de ChromaDB'
    )
    query_parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='Nombre de la colección'
    )
    query_parser.add_argument(
        '--n-results',
        type=int,
        default=5,
        help='Número de resultados'
    )
    
    # Comando: stats
    stats_parser = subparsers.add_parser('stats', help='Mostrar estadísticas')
    stats_parser.add_argument(
        '--persist-dir',
        type=str,
        default='output_rag/chroma_db',
        help='Directorio de ChromaDB'
    )
    stats_parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='Nombre de la colección'
    )
    
    # Comando: reset
    reset_parser = subparsers.add_parser('reset', help='Resetear colección')
    reset_parser.add_argument(
        '--persist-dir',
        type=str,
        default='output_rag/chroma_db',
        help='Directorio de ChromaDB'
    )
    reset_parser.add_argument(
        '--collection',
        type=str,
        default='documents',
        help='Nombre de la colección'
    )
    
    args = parser.parse_args()
    
    if args.command == 'load':
        # Cargar embeddings
        store = VectorStore(
            persist_directory=args.persist_dir,
            collection_name=args.collection
        )
        
        stats = store.add_embeddings_from_directory(args.embeddings_dir)
        
        print("\n✓ Carga completada")
        print(f"  Chunks: {stats['chunks_loaded']}")
        print(f"  Total: {stats['total_documents']}")
    
    elif args.command == 'query':
        # Buscar
        store = VectorStore(
            persist_directory=args.persist_dir,
            collection_name=args.collection
        )
        
        # Cargar embedding de consulta
        with open(args.query_embedding_file, 'r') as f:
            query_data = json.load(f)
            query_embedding = query_data.get('embedding', [])
        
        results = store.query_by_embedding(
            query_embedding=query_embedding,
            n_results=args.n_results
        )
        
        print(f"\n🔍 Resultados de búsqueda ({results['n_results']}):")
        for i, result in enumerate(results['results'], 1):
            print(f"\n{i}. {result['id']} (similarity: {result['similarity']:.4f})")
            print(f"   {result['document'][:100]}...")
    
    elif args.command == 'stats':
        # Estadísticas
        store = VectorStore(
            persist_directory=args.persist_dir,
            collection_name=args.collection
        )
        
        stats = store.get_stats()
        
        print("\n📊 Estadísticas:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif args.command == 'reset':
        # Resetear
        store = VectorStore(
            persist_directory=args.persist_dir,
            collection_name=args.collection
        )
        
        confirm = input(f"¿Seguro que quieres resetear '{args.collection}'? (s/N): ")
        if confirm.lower() == 's':
            store.reset_collection()
        else:
            print("Operación cancelada")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
