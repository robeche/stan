"""
Ejemplos de Reranking en Sistema RAG
=====================================

Demuestra el uso del reranker integrado con búsqueda vectorial.
"""

from vector_store import VectorStore
from embedding_generator import EmbeddingGenerator
from reranker import Reranker, RerankResult


def ejemplo_1_comparacion_basica():
    """Ejemplo 1: Comparar resultados con y sin reranking."""
    print("\n" + "="*60)
    print("EJEMPLO 1: COMPARACIÓN CON vs SIN RERANKING")
    print("="*60)
    
    # Inicializar componentes
    print("\n📦 Inicializando componentes...")
    
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    generator = EmbeddingGenerator("bge-m3")
    reranker = Reranker("bge-reranker-v2-m3")
    
    # Query de prueba
    query = "What are the blade design specifications?"
    
    print(f"\n🔍 Query: {query}")
    
    # 1. Búsqueda vectorial sin reranking
    print("\n" + "-"*60)
    print("📊 RESULTADOS SIN RERANKING (top 5):")
    
    query_embedding = generator.generate_embedding(query)
    search_results = store.query_by_embedding(
        query_embedding=query_embedding,
        n_results=10  # Recuperar más para reranking
    )
    
    for i, result in enumerate(search_results['results'][:5], 1):
        print(f"\n{i}. {result['id']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Content: {result['document'][:100]}...")
    
    # 2. Con reranking
    print("\n" + "-"*60)
    print("🎯 RESULTADOS CON RERANKING (top 5):")
    
    reranked = reranker.rerank_results(
        query=query,
        search_results=search_results,
        top_k=5,
        return_full_results=True
    )
    
    for i, result in enumerate(reranked, 1):
        change_symbol = "↑" if result.rank_change > 0 else "↓" if result.rank_change < 0 else "="
        
        print(f"\n{i}. {result.id} {change_symbol}{abs(result.rank_change)}")
        print(f"   Rerank Score: {result.rerank_score:.4f}")
        print(f"   Original Similarity: {result.original_score:.4f}")
        print(f"   Content: {result.document[:100]}...")


def ejemplo_2_mejora_precision():
    """Ejemplo 2: Mostrar mejora de precisión con reranking."""
    print("\n" + "="*60)
    print("EJEMPLO 2: MEJORA DE PRECISIÓN")
    print("="*60)
    
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    generator = EmbeddingGenerator("bge-m3")
    reranker = Reranker("bge-reranker-v2-m3")
    
    # Queries que pueden beneficiarse del reranking
    queries = [
        "tower height specifications",
        "rotor diameter measurements",
        "power generation capacity"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print('='*60)
        
        # Búsqueda vectorial
        query_embedding = generator.generate_embedding(query)
        search_results = store.query_by_embedding(
            query_embedding=query_embedding,
            n_results=10
        )
        
        # Reranking
        reranked = reranker.rerank_results(
            query=query,
            search_results=search_results,
            top_k=3,
            return_full_results=True
        )
        
        # Mostrar top 3
        print(f"\n📊 Top 3 después de reranking:")
        for i, result in enumerate(reranked, 1):
            improvement = result.rerank_score - result.original_score
            print(f"\n{i}. {result.id}")
            print(f"   Rerank: {result.rerank_score:.4f} | "
                  f"Original: {result.original_score:.4f} | "
                  f"Mejora: {improvement:+.4f}")
            print(f"   {result.document[:80]}...")


def ejemplo_3_analisis_cambios():
    """Ejemplo 3: Análisis detallado de cambios de ranking."""
    print("\n" + "="*60)
    print("EJEMPLO 3: ANÁLISIS DE CAMBIOS DE RANKING")
    print("="*60)
    
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    generator = EmbeddingGenerator("bge-m3")
    reranker = Reranker("bge-reranker-v2-m3")
    
    query = "What are the technical specifications?"
    
    print(f"\n🔍 Query: {query}")
    
    # Búsqueda inicial
    query_embedding = generator.generate_embedding(query)
    search_results = store.query_by_embedding(
        query_embedding=query_embedding,
        n_results=15
    )
    
    # Reranking
    reranked = reranker.rerank_results(
        query=query,
        search_results=search_results,
        top_k=10,
        return_full_results=True
    )
    
    # Analizar cambios
    print(f"\n📊 Análisis de cambios:")
    
    moved_up = [r for r in reranked if r.rank_change > 0]
    moved_down = [r for r in reranked if r.rank_change < 0]
    unchanged = [r for r in reranked if r.rank_change == 0]
    
    print(f"\n  Subieron: {len(moved_up)}")
    print(f"  Bajaron: {len(moved_down)}")
    print(f"  Sin cambio: {len(unchanged)}")
    
    if moved_up:
        print(f"\n📈 Documentos que más subieron:")
        top_movers = sorted(moved_up, key=lambda x: x.rank_change, reverse=True)[:3]
        for result in top_movers:
            print(f"\n  {result.id} (↑{result.rank_change} posiciones)")
            print(f"  Rerank: {result.rerank_score:.4f} | Original: {result.original_score:.4f}")
            print(f"  {result.document[:80]}...")
    
    if moved_down:
        print(f"\n📉 Documentos que más bajaron:")
        bottom_movers = sorted(moved_down, key=lambda x: x.rank_change)[:3]
        for result in bottom_movers:
            print(f"\n  {result.id} (↓{abs(result.rank_change)} posiciones)")
            print(f"  Rerank: {result.rerank_score:.4f} | Original: {result.original_score:.4f}")
            print(f"  {result.document[:80]}...")


def ejemplo_4_contexto_rag():
    """Ejemplo 4: Construcción de contexto para LLM."""
    print("\n" + "="*60)
    print("EJEMPLO 4: CONSTRUCCIÓN DE CONTEXTO PARA LLM")
    print("="*60)
    
    store = VectorStore(
        persist_directory="output_rag/chroma_db",
        collection_name="nrel5mw_docs"
    )
    
    generator = EmbeddingGenerator("bge-m3")
    reranker = Reranker("bge-reranker-v2-m3")
    
    query = "Explain the wind turbine blade design and specifications"
    
    print(f"\n🔍 Query: {query}")
    
    # Pipeline completo
    query_embedding = generator.generate_embedding(query)
    
    # Recuperar candidatos
    search_results = store.query_by_embedding(
        query_embedding=query_embedding,
        n_results=15
    )
    
    # Reranking para obtener los mejores
    reranked = reranker.rerank_results(
        query=query,
        search_results=search_results,
        top_k=5,
        return_full_results=True
    )
    
    # Construir contexto para LLM
    print(f"\n📄 CONTEXTO PARA LLM (Top 5):")
    print("="*60)
    
    context_parts = []
    for i, result in enumerate(reranked, 1):
        context_parts.append(f"\n--- Document {i} (Score: {result.rerank_score:.4f}) ---")
        context_parts.append(result.document)
    
    context = "\n".join(context_parts)
    
    print(context[:1000] + "...\n")
    
    print(f"\n📊 Estadísticas del contexto:")
    print(f"  Documentos: {len(reranked)}")
    print(f"  Tokens aprox: {len(context.split())}")
    print(f"  Caracteres: {len(context)}")


def menu_interactivo():
    """Menú interactivo para ejecutar ejemplos."""
    ejemplos = {
        "1": ("Comparación con/sin reranking", ejemplo_1_comparacion_basica),
        "2": ("Mejora de precisión", ejemplo_2_mejora_precision),
        "3": ("Análisis de cambios", ejemplo_3_analisis_cambios),
        "4": ("Construcción contexto LLM", ejemplo_4_contexto_rag),
        "5": ("Ejecutar todos", None),
    }
    
    print("\n" + "="*60)
    print("EJEMPLOS DE RERANKING")
    print("="*60)
    
    for key, (desc, _) in ejemplos.items():
        print(f"{key}. {desc}")
    
    print("0. Salir")
    print("="*60)
    
    opcion = input("\nSelecciona un ejemplo: ").strip()
    
    if opcion == "0":
        print("¡Hasta luego!")
        return
    
    if opcion == "5":
        # Ejecutar todos
        for key, (_, func) in ejemplos.items():
            if key != "5" and func:
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
            "1": ejemplo_1_comparacion_basica,
            "2": ejemplo_2_mejora_precision,
            "3": ejemplo_3_analisis_cambios,
            "4": ejemplo_4_contexto_rag,
        }
        
        if ejemplo_num in ejemplos_map:
            ejemplos_map[ejemplo_num]()
        else:
            print(f"Uso: python {sys.argv[0]} [1-4]")
    else:
        # Modo interactivo
        menu_interactivo()
