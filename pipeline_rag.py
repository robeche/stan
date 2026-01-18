"""
🔄 Pipeline de Integración RAG

Este script muestra cómo integrar el módulo de chunking
con un sistema RAG completo (Retrieval-Augmented Generation).

Incluye ejemplos de:
- Generación de embeddings
- Almacenamiento en vector database
- Búsqueda semántica
- Generación de respuestas
"""

from document_chunker import DocumentChunker, ChunkConfig, ChunkingStrategy
import json
import os


# ==============================================================================
# PASO 1: CHUNKING DEL DOCUMENTO
# ==============================================================================

def paso_1_chunking(documento_path, output_dir):
    """
    Divide el documento en chunks optimizados para RAG.
    
    Args:
        documento_path: Ruta al documento Markdown
        output_dir: Directorio donde guardar los chunks
        
    Returns:
        Lista de chunks generados
    """
    print("\n" + "="*70)
    print("PASO 1: CHUNKING DEL DOCUMENTO")
    print("="*70)
    
    # Configuración óptima para RAG
    config = ChunkConfig(
        chunk_size=1200,        # Tamaño óptimo para embeddings
        chunk_overlap=200,      # Overlap para mantener contexto
        min_chunk_size=300,     # Evitar chunks muy pequeños
        max_chunk_size=2500,    # Límite superior
        strategy=ChunkingStrategy.HYBRID,
        preserve_tables=True,
        include_metadata=True
    )
    
    # Crear chunker y procesar
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(documento_path)
    
    # Guardar chunks
    os.makedirs(output_dir, exist_ok=True)
    chunker.save_chunks(output_dir, format='json')
    
    print(f"\n✓ {len(chunks)} chunks generados y guardados")
    return chunks


# ==============================================================================
# PASO 2: GENERACIÓN DE EMBEDDINGS (SIMULADO)
# ==============================================================================

def paso_2_generar_embeddings(chunks):
    """
    Genera embeddings para cada chunk.
    
    En un sistema real, aquí usarías:
    - OpenAI Embeddings API
    - Hugging Face Sentence Transformers
    - Cohere Embeddings
    - etc.
    
    Args:
        chunks: Lista de chunks
        
    Returns:
        Lista de diccionarios con chunks y embeddings
    """
    print("\n" + "="*70)
    print("PASO 2: GENERACIÓN DE EMBEDDINGS")
    print("="*70)
    print("(Simulando generación de embeddings...)")
    
    chunks_con_embeddings = []
    
    for chunk in chunks:
        # SIMULACIÓN: En un sistema real, aquí generarías el embedding real
        # Ejemplo con OpenAI:
        # response = openai.Embedding.create(
        #     input=chunk.content,
        #     model="text-embedding-ada-002"
        # )
        # embedding = response['data'][0]['embedding']
        
        # Por ahora, simulamos con un vector vacío
        embedding_simulado = [0.0] * 1536  # Dimensión típica de embeddings
        
        chunks_con_embeddings.append({
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'metadata': chunk.metadata,
            'embedding': embedding_simulado
        })
    
    print(f"✓ Embeddings generados para {len(chunks_con_embeddings)} chunks")
    print(f"  Dimensión: {len(chunks_con_embeddings[0]['embedding'])}")
    
    return chunks_con_embeddings


# ==============================================================================
# PASO 3: ALMACENAMIENTO EN VECTOR DATABASE (SIMULADO)
# ==============================================================================

def paso_3_almacenar_en_vectordb(chunks_con_embeddings, output_file):
    """
    Almacena chunks y embeddings en una base de datos vectorial.
    
    En un sistema real, aquí usarías:
    - Pinecone
    - Weaviate
    - Milvus
    - Chroma
    - FAISS
    - etc.
    
    Args:
        chunks_con_embeddings: Chunks con sus embeddings
        output_file: Archivo donde simular el almacenamiento
    """
    print("\n" + "="*70)
    print("PASO 3: ALMACENAMIENTO EN VECTOR DATABASE")
    print("="*70)
    print("(Simulando almacenamiento en vector database...)")
    
    # Preparar datos para almacenamiento
    # En un sistema real, aquí insertarías en tu vector DB
    datos_para_db = []
    
    for item in chunks_con_embeddings:
        datos_para_db.append({
            'id': f"chunk_{item['chunk_id']:04d}",
            'content': item['content'],
            'metadata': item['metadata'],
            # En sistema real, aquí iría el embedding completo
            'embedding_preview': item['embedding'][:5]  # Solo primeros 5 valores
        })
    
    # Guardar simulación
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(datos_para_db, f, ensure_ascii=False, indent=2)
    
    print(f"✓ {len(datos_para_db)} chunks almacenados (simulado)")
    print(f"  Archivo: {output_file}")


# ==============================================================================
# PASO 4: BÚSQUEDA SEMÁNTICA (SIMULADO)
# ==============================================================================

def paso_4_busqueda_semantica(query, chunks_con_embeddings, top_k=3):
    """
    Realiza búsqueda semántica en los chunks.
    
    En un sistema real:
    1. Generarías embedding de la query
    2. Buscarías en vector DB los chunks más similares
    3. Retornarías los top_k resultados
    
    Args:
        query: Pregunta del usuario
        chunks_con_embeddings: Chunks disponibles
        top_k: Número de chunks a retornar
        
    Returns:
        Lista de chunks más relevantes
    """
    print("\n" + "="*70)
    print("PASO 4: BÚSQUEDA SEMÁNTICA")
    print("="*70)
    print(f"Query: '{query}'")
    print(f"(Simulando búsqueda en vector database...)")
    
    # SIMULACIÓN: Búsqueda por palabras clave simple
    # En sistema real, sería búsqueda por similaridad de embeddings
    
    query_lower = query.lower()
    chunks_con_score = []
    
    for chunk in chunks_con_embeddings:
        content_lower = chunk['content'].lower()
        
        # Score simple basado en palabras clave
        score = 0
        for word in query_lower.split():
            if word in content_lower:
                score += content_lower.count(word)
        
        if score > 0:
            chunks_con_score.append({
                'chunk': chunk,
                'score': score
            })
    
    # Ordenar por score y tomar top_k
    chunks_con_score.sort(key=lambda x: x['score'], reverse=True)
    resultados = chunks_con_score[:top_k]
    
    print(f"\n✓ {len(resultados)} chunks más relevantes encontrados:")
    for i, resultado in enumerate(resultados, 1):
        chunk_id = resultado['chunk']['chunk_id']
        score = resultado['score']
        page = resultado['chunk']['metadata'].get('page', 'N/A')
        print(f"  {i}. Chunk #{chunk_id} (Página {page}) - Score: {score}")
    
    return [r['chunk'] for r in resultados]


# ==============================================================================
# PASO 5: GENERACIÓN DE RESPUESTA CON CONTEXTO
# ==============================================================================

def paso_5_generar_respuesta(query, chunks_relevantes):
    """
    Genera respuesta usando los chunks relevantes como contexto.
    
    En un sistema real, aquí usarías:
    - OpenAI GPT-4/GPT-3.5
    - Anthropic Claude
    - Llama
    - etc.
    
    Args:
        query: Pregunta del usuario
        chunks_relevantes: Chunks recuperados
        
    Returns:
        Respuesta generada (simulada)
    """
    print("\n" + "="*70)
    print("PASO 5: GENERACIÓN DE RESPUESTA")
    print("="*70)
    print("(Simulando generación con LLM...)")
    
    # Construir contexto
    contexto = "\n\n---\n\n".join([
        f"[Chunk {chunk['chunk_id']}, Página {chunk['metadata'].get('page', 'N/A')}]\n{chunk['content']}"
        for chunk in chunks_relevantes
    ])
    
    # En un sistema real, aquí construirías el prompt y llamarías al LLM:
    # prompt = f"""
    # Contexto:
    # {contexto}
    # 
    # Pregunta: {query}
    # 
    # Responde basándote solo en el contexto proporcionado.
    # """
    # 
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # 
    # respuesta = response.choices[0].message.content
    
    # SIMULACIÓN
    print(f"\nPregunta: {query}")
    print("\nContexto utilizado:")
    print("-" * 70)
    print(contexto[:500] + "...\n" if len(contexto) > 500 else contexto)
    print("-" * 70)
    
    respuesta_simulada = f"""
    [RESPUESTA SIMULADA]
    
    Basándome en los {len(chunks_relevantes)} fragmentos relevantes encontrados 
    en el documento, puedo responder a tu pregunta sobre: {query}
    
    Los chunks relevantes provienen de:
    {', '.join([f"Página {c['metadata'].get('page', 'N/A')}" for c in chunks_relevantes])}
    
    En un sistema real con LLM, aquí aparecería una respuesta 
    contextualizada y precisa basada en el contenido específico 
    de los fragmentos recuperados.
    """
    
    print("\nRespuesta generada:")
    print(respuesta_simulada)
    
    return respuesta_simulada


# ==============================================================================
# PIPELINE COMPLETO
# ==============================================================================

def pipeline_rag_completo(documento_path, query, output_dir="output_rag"):
    """
    Ejecuta el pipeline RAG completo desde chunking hasta respuesta.
    
    Args:
        documento_path: Ruta al documento fuente
        query: Pregunta del usuario
        output_dir: Directorio para archivos intermedios
        
    Returns:
        Respuesta generada
    """
    print("\n" + "="*70)
    print("🚀 PIPELINE RAG COMPLETO")
    print("="*70)
    print(f"Documento: {documento_path}")
    print(f"Query: '{query}'")
    print("="*70)
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # PASO 1: Chunking
    chunks = paso_1_chunking(documento_path, os.path.join(output_dir, "chunks"))
    
    # PASO 2: Generar embeddings
    chunks_con_embeddings = paso_2_generar_embeddings(chunks)
    
    # PASO 3: Almacenar en vector DB
    paso_3_almacenar_en_vectordb(
        chunks_con_embeddings,
        os.path.join(output_dir, "vector_db_simulation.json")
    )
    
    # PASO 4: Búsqueda semántica
    chunks_relevantes = paso_4_busqueda_semantica(query, chunks_con_embeddings, top_k=3)
    
    # PASO 5: Generar respuesta
    respuesta = paso_5_generar_respuesta(query, chunks_relevantes)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETADO")
    print("="*70)
    
    return respuesta


# ==============================================================================
# EJEMPLOS DE QUERIES
# ==============================================================================

def ejecutar_ejemplos():
    """Ejecuta varios ejemplos de queries sobre el documento"""
    
    documento = "output_simple/NREL5MW_Reduced/documento_concatenado.md"
    
    queries = [
        "What is the power rating of the NREL baseline wind turbine?",
        "What are the main properties of the REpower 5M machine?",
        "What is the rotor diameter and hub height?",
        "Tell me about the blade structural properties"
    ]
    
    print("\n" + "="*70)
    print("🎯 EJECUTANDO MÚLTIPLES QUERIES")
    print("="*70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"QUERY {i}/{len(queries)}")
        print(f"{'='*70}")
        
        pipeline_rag_completo(
            documento_path=documento,
            query=query,
            output_dir=f"output_rag/query_{i}"
        )
        
        if i < len(queries):
            input("\nPresiona Enter para continuar con la siguiente query...")


# ==============================================================================
# EJEMPLO DE INTEGRACIÓN CON API REAL
# ==============================================================================

def ejemplo_integracion_openai():
    """
    Ejemplo de cómo integrar con OpenAI API (código comentado).
    Descomenta y añade tu API key para usar.
    """
    codigo_ejemplo = '''
# Ejemplo de integración real con OpenAI
import openai

# Configurar API key
openai.api_key = "tu-api-key-aqui"

# Paso 1: Chunking (igual que antes)
from document_chunker import DocumentChunker, ChunkConfig
chunker = DocumentChunker(ChunkConfig(chunk_size=1200))
chunks = chunker.chunk_document("documento.md")

# Paso 2: Generar embeddings REALES
embeddings = []
for chunk in chunks:
    response = openai.Embedding.create(
        input=chunk.content,
        model="text-embedding-ada-002"
    )
    embeddings.append({
        'chunk_id': chunk.chunk_id,
        'content': chunk.content,
        'embedding': response['data'][0]['embedding'],
        'metadata': chunk.metadata
    })

# Paso 3: Almacenar en Pinecone (ejemplo)
import pinecone
pinecone.init(api_key="tu-pinecone-key", environment="us-west1-gcp")
index = pinecone.Index("tu-index")

vectors_to_upsert = [
    (f"chunk_{e['chunk_id']}", e['embedding'], e['metadata'])
    for e in embeddings
]
index.upsert(vectors=vectors_to_upsert)

# Paso 4: Query con búsqueda semántica REAL
query = "What is the power rating?"
query_embedding = openai.Embedding.create(
    input=query,
    model="text-embedding-ada-002"
)['data'][0]['embedding']

results = index.query(query_embedding, top_k=3, include_metadata=True)

# Paso 5: Generar respuesta con GPT-4
context = "\\n\\n".join([match['metadata']['content'] for match in results['matches']])
prompt = f"""
Contexto:
{context}

Pregunta: {query}

Responde basándote solo en el contexto.
"""

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
'''
    
    print("\n" + "="*70)
    print("💡 EJEMPLO DE INTEGRACIÓN CON OpenAI")
    print("="*70)
    print("\nCódigo de ejemplo para integración real:")
    print(codigo_ejemplo)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "ejemplos":
            ejecutar_ejemplos()
        elif sys.argv[1] == "openai":
            ejemplo_integracion_openai()
        else:
            # Query personalizada
            query = " ".join(sys.argv[1:])
            pipeline_rag_completo(
                documento_path="output_simple/NREL5MW_Reduced/documento_concatenado.md",
                query=query,
                output_dir="output_rag/custom_query"
            )
    else:
        # Ejecutar un ejemplo por defecto
        pipeline_rag_completo(
            documento_path="output_simple/NREL5MW_Reduced/documento_concatenado.md",
            query="What is the NREL baseline wind turbine?",
            output_dir="output_rag/default"
        )
