"""
Ejemplos de uso del Vector Store con ChromaDB
==============================================
"""

from vector_store import VectorStore
from embedding_generator import EmbeddingGenerator
from pathlib import Path
import json
import numpy as np


def ejemplo_1_cargar_embeddings():
    """Ejemplo 1: Cargar embeddings en ChromaDB."""
    print("\n" + "="*60)
    print("EJEMPLO 1: Cargar Embeddings en ChromaDB")
    print("="*60)
    
    # Crear vector store
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    # Cargar embeddings desde directorio
    stats = store.add_embeddings_from_directory(
        embeddings_dir="output_rag/embeddings"
    )
    
    print(f"\n✓ Carga completada")
    print(f"  Chunks cargados: {stats['chunks_loaded']}")
    print(f"  Total documentos: {stats['total_documents']}")


def ejemplo_2_busqueda_basica():
    """Ejemplo 2: Búsqueda básica por similitud."""
    print("\n" + "="*60)
    print("EJEMPLO 2: Búsqueda por Similitud")
    print("="*60)
    
    # Inicializar
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    generator = EmbeddingGenerator("bge-m3")
    
    # Query de ejemplo
    query = "What is the blade design of the wind turbine?"
    print(f"\n🔍 Query: {query}")
    
    # Generar embedding para la query
    query_embedding = generator.generate_embedding(query)
    
    # Buscar documentos similares
    results = store.query_by_embedding(
        query_embedding=query_embedding,
        n_results=3
    )
    
    print(f"\n📊 Resultados encontrados: {results['n_results']}")
    
    for i, result in enumerate(results['results'], 1):
        print(f"\n{i}. {result['id']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Distance: {result['distance']:.4f}")
        print(f"   Content: {result['document'][:150]}...")
        print(f"   Metadata: chunk_id={result['metadata'].get('chunk_id')}, "
              f"length={result['metadata'].get('length')}")


def ejemplo_3_busqueda_con_filtros():
    """Ejemplo 3: Búsqueda con filtros de metadata."""
    print("\n" + "="*60)
    print("EJEMPLO 3: Búsqueda con Filtros")
    print("="*60)
    
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    generator = EmbeddingGenerator("bge-m3")
    
    # Query
    query = "turbine specifications"
    query_embedding = generator.generate_embedding(query)
    
    # Buscar solo en chunks largos (>500 caracteres)
    print(f"\n🔍 Query: {query}")
    print(f"📋 Filtro: length > 500")
    
    results = store.query_by_embedding(
        query_embedding=query_embedding,
        n_results=3,
        where={"length": {"$gt": 500}}
    )
    
    print(f"\n📊 Resultados: {results['n_results']}")
    
    for i, result in enumerate(results['results'], 1):
        print(f"\n{i}. {result['id']} (similarity: {result['similarity']:.4f})")
        print(f"   Length: {result['metadata']['length']}")
        print(f"   Content: {result['document'][:100]}...")


def ejemplo_4_busqueda_por_contenido():
    """Ejemplo 4: Buscar documentos que contengan palabras específicas."""
    print("\n" + "="*60)
    print("EJEMPLO 4: Búsqueda por Contenido")
    print("="*60)
    
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    generator = EmbeddingGenerator("bge-m3")
    
    # Buscar documentos que contengan "blade"
    query = "rotor design characteristics"
    query_embedding = generator.generate_embedding(query)
    
    print(f"\n🔍 Query: {query}")
    print(f"📋 Filtro: documento debe contener 'blade'")
    
    results = store.query_by_embedding(
        query_embedding=query_embedding,
        n_results=3,
        where_document={"$contains": "blade"}
    )
    
    print(f"\n📊 Resultados: {results['n_results']}")
    
    for i, result in enumerate(results['results'], 1):
        print(f"\n{i}. {result['id']} (similarity: {result['similarity']:.4f})")
        print(f"   Content: {result['document'][:120]}...")


def ejemplo_5_obtener_por_ids():
    """Ejemplo 5: Obtener chunks específicos por ID."""
    print("\n" + "="*60)
    print("EJEMPLO 5: Obtener por IDs")
    print("="*60)
    
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    # Obtener chunks específicos
    ids = ["chunk_0005", "chunk_0010", "chunk_0015"]
    
    print(f"\n📋 Obteniendo chunks: {ids}")
    
    results = store.get_by_ids(ids)
    
    print(f"\n📊 Documentos obtenidos: {len(results['ids'])}")
    
    for i, (id, doc, meta) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
        print(f"\n{i+1}. {id}")
        print(f"   Length: {meta['length']}")
        print(f"   Content: {doc[:100]}...")


def ejemplo_6_estadisticas():
    """Ejemplo 6: Ver estadísticas del vector store."""
    print("\n" + "="*60)
    print("EJEMPLO 6: Estadísticas")
    print("="*60)
    
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    # Obtener estadísticas
    stats = store.get_stats()
    
    print(f"\n📊 Estadísticas de la colección:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Ver muestra de documentos
    print(f"\n📄 Muestra de documentos:")
    peek = store.peek(limit=3)
    
    for i, (id, doc) in enumerate(zip(peek['ids'], peek['documents'])):
        print(f"\n{i+1}. {id}")
        print(f"   {doc[:80]}...")


def ejemplo_7_multiples_queries():
    """Ejemplo 7: Múltiples queries para construir contexto RAG."""
    print("\n" + "="*60)
    print("EJEMPLO 7: Múltiples Queries para RAG")
    print("="*60)
    
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    generator = EmbeddingGenerator("bge-m3")
    
    # Varias preguntas relacionadas
    queries = [
        "What are the blade specifications?",
        "How tall is the tower?",
        "What is the rotor diameter?",
        "Power generation capacity"
    ]
    
    print("\n🔍 Ejecutando múltiples queries...")
    
    all_chunks = {}
    
    for query in queries:
        print(f"\n• Query: {query}")
        
        query_embedding = generator.generate_embedding(query)
        results = store.query_by_embedding(
            query_embedding=query_embedding,
            n_results=2
        )
        
        # Recopilar chunks únicos
        for result in results['results']:
            chunk_id = result['id']
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = {
                    'content': result['document'],
                    'similarity': result['similarity'],
                    'queries': [query]
                }
            else:
                all_chunks[chunk_id]['queries'].append(query)
                # Mantener la mejor similitud
                all_chunks[chunk_id]['similarity'] = max(
                    all_chunks[chunk_id]['similarity'],
                    result['similarity']
                )
    
    # Mostrar chunks consolidados
    print(f"\n📊 Chunks únicos encontrados: {len(all_chunks)}")
    
    sorted_chunks = sorted(
        all_chunks.items(),
        key=lambda x: x[1]['similarity'],
        reverse=True
    )
    
    print(f"\n📄 Top 5 chunks para contexto RAG:")
    for i, (chunk_id, data) in enumerate(sorted_chunks[:5], 1):
        print(f"\n{i}. {chunk_id} (similarity: {data['similarity']:.4f})")
        print(f"   Relevante para: {', '.join(data['queries'])}")
        print(f"   Content: {data['content'][:100]}...")


def menu_interactivo():
    """Menú interactivo para ejecutar ejemplos."""
    ejemplos = {
        "1": ("Cargar embeddings en ChromaDB", ejemplo_1_cargar_embeddings),
        "2": ("Búsqueda básica", ejemplo_2_busqueda_basica),
        "3": ("Búsqueda con filtros", ejemplo_3_busqueda_con_filtros),
        "4": ("Búsqueda por contenido", ejemplo_4_busqueda_por_contenido),
        "5": ("Obtener por IDs", ejemplo_5_obtener_por_ids),
        "6": ("Ver estadísticas", ejemplo_6_estadisticas),
        "7": ("Múltiples queries (RAG)", ejemplo_7_multiples_queries),
        "8": ("Ejecutar todos", None),
    }
    
    print("\n" + "="*60)
    print("EJEMPLOS DE VECTOR STORE (ChromaDB)")
    print("="*60)
    
    for key, (desc, _) in ejemplos.items():
        print(f"{key}. {desc}")
    
    print("0. Salir")
    print("="*60)
    
    opcion = input("\nSelecciona un ejemplo: ").strip()
    
    if opcion == "0":
        print("¡Hasta luego!")
        return
    
    if opcion == "8":
        # Ejecutar todos
        for key, (_, func) in ejemplos.items():
            if key != "8" and func:
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ Error en ejemplo {key}: {e}")
        return
    
    if opcion in ejemplos and ejemplos[opcion][1]:
        try:
            ejemplos[opcion][1]()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Opción inválida")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Modo línea de comandos
        ejemplo_num = sys.argv[1]
        
        ejemplos_map = {
            "1": ejemplo_1_cargar_embeddings,
            "2": ejemplo_2_busqueda_basica,
            "3": ejemplo_3_busqueda_con_filtros,
            "4": ejemplo_4_busqueda_por_contenido,
            "5": ejemplo_5_obtener_por_ids,
            "6": ejemplo_6_estadisticas,
            "7": ejemplo_7_multiples_queries,
        }
        
        if ejemplo_num in ejemplos_map:
            ejemplos_map[ejemplo_num]()
        else:
            print(f"Uso: python {sys.argv[0]} [1-7]")
    else:
        # Modo interactivo
        menu_interactivo()
