"""
📘 Ejemplos de Uso del Document Chunker

Este script demuestra diferentes formas de usar el módulo de chunking
para preparar documentos para sistemas RAG.
"""

from document_chunker import DocumentChunker, ChunkConfig, ChunkingStrategy, create_chunks_from_file


def ejemplo_basico():
    """Ejemplo básico: configuración por defecto"""
    print("\n" + "="*70)
    print("EJEMPLO 1: USO BÁSICO")
    print("="*70)
    
    config = ChunkConfig()
    chunker = DocumentChunker(config)
    
    chunks = chunker.chunk_document(
        "output_simple/NREL5MW_Reduced/documento_concatenado.md"
    )
    
    # Mostrar primeros 3 chunks
    for chunk in chunks[:3]:
        print(f"\n{chunk}")
        print(f"Metadatos: {chunk.metadata}")
        print(f"Preview: {chunk.content[:100]}...")


def ejemplo_configuracion_personalizada():
    """Ejemplo con configuración personalizada para RAG conversacional"""
    print("\n" + "="*70)
    print("EJEMPLO 2: CONFIGURACIÓN PERSONALIZADA PARA RAG CONVERSACIONAL")
    print("="*70)
    
    config = ChunkConfig(
        chunk_size=800,         # Chunks más pequeños para respuestas rápidas
        chunk_overlap=150,      # Overlap moderado
        min_chunk_size=200,
        max_chunk_size=1500,
        strategy=ChunkingStrategy.HYBRID
    )
    
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(
        "output_simple/NREL5MW_Reduced/documento_concatenado.md"
    )
    
    print(f"\n✓ Generados {len(chunks)} chunks optimizados para RAG conversacional")


def ejemplo_semantic():
    """Ejemplo con estrategia semántica para preservar coherencia"""
    print("\n" + "="*70)
    print("EJEMPLO 3: ESTRATEGIA SEMÁNTICA (MÁXIMA COHERENCIA)")
    print("="*70)
    
    config = ChunkConfig(
        chunk_size=1000,
        max_chunk_size=3000,    # Permitir secciones completas
        strategy=ChunkingStrategy.SEMANTIC,
        preserve_tables=True
    )
    
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(
        "output_simple/NREL5MW_Reduced/documento_concatenado.md"
    )
    
    # Analizar distribución de páginas
    paginas = {}
    for chunk in chunks:
        page = chunk.metadata.get('page', 'N/A')
        paginas[page] = paginas.get(page, 0) + 1
    
    print("\nDistribución de chunks por página:")
    for page, count in sorted(paginas.items()):
        print(f"  Página {page}: {count} chunks")


def ejemplo_analisis_chunks():
    """Ejemplo de análisis de chunks generados"""
    print("\n" + "="*70)
    print("EJEMPLO 4: ANÁLISIS DETALLADO DE CHUNKS")
    print("="*70)
    
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(
        "output_simple/NREL5MW_Reduced/documento_concatenado.md"
    )
    
    # Encontrar chunk más largo
    chunk_mas_largo = max(chunks, key=lambda c: len(c.content))
    print(f"\n📏 Chunk más largo: #{chunk_mas_largo.chunk_id}")
    print(f"   Longitud: {len(chunk_mas_largo.content)} caracteres")
    print(f"   Página: {chunk_mas_largo.metadata.get('page', 'N/A')}")
    print(f"   Sección: {chunk_mas_largo.metadata.get('section_title', 'N/A')}")
    
    # Encontrar chunk más corto (excluyendo muy pequeños)
    chunks_significativos = [c for c in chunks if len(c.content) > 100]
    if chunks_significativos:
        chunk_mas_corto = min(chunks_significativos, key=lambda c: len(c.content))
        print(f"\n📐 Chunk más corto: #{chunk_mas_corto.chunk_id}")
        print(f"   Longitud: {len(chunk_mas_corto.content)} caracteres")
        print(f"   Página: {chunk_mas_corto.metadata.get('page', 'N/A')}")
    
    # Chunks con tablas (buscar chunks que mencionen tablas)
    chunks_con_tablas = [
        c for c in chunks 
        if '|' in c.content and '---' in c.content
    ]
    print(f"\n📊 Chunks con tablas: {len(chunks_con_tablas)}")
    
    # Chunks por estrategia
    estrategias = {}
    for chunk in chunks:
        strategy = chunk.metadata.get('strategy', 'unknown')
        estrategias[strategy] = estrategias.get(strategy, 0) + 1
    
    print("\n🎯 Chunks por estrategia:")
    for strategy, count in estrategias.items():
        print(f"   {strategy}: {count} chunks")


def ejemplo_funcion_rapida():
    """Ejemplo usando la función de conveniencia"""
    print("\n" + "="*70)
    print("EJEMPLO 5: FUNCIÓN RÁPIDA (SHORTCUT)")
    print("="*70)
    
    chunks = create_chunks_from_file(
        input_file="output_simple/NREL5MW_Reduced/documento_concatenado.md",
        output_dir="output_simple/NREL5MW_Reduced/chunks_ejemplo",
        chunk_size=1000,
        overlap=200,
        strategy='hybrid'
    )
    
    print(f"\n✓ Función rápida completada: {len(chunks)} chunks generados")


def ejemplo_exportar_para_embedding():
    """Ejemplo: preparar chunks para generar embeddings"""
    print("\n" + "="*70)
    print("EJEMPLO 6: PREPARAR CHUNKS PARA EMBEDDINGS")
    print("="*70)
    
    # Configuración óptima para modelos de embedding (512 tokens)
    MAX_TOKENS = 512
    CHARS_PER_TOKEN = 4  # Aproximación para inglés
    
    config = ChunkConfig(
        chunk_size=int(MAX_TOKENS * CHARS_PER_TOKEN * 0.8),  # ~1600 chars
        chunk_overlap=200,
        strategy=ChunkingStrategy.HYBRID
    )
    
    chunker = DocumentChunker(config)
    chunks = chunker.chunk_document(
        "output_simple/NREL5MW_Reduced/documento_concatenado.md"
    )
    
    # Simular preparación para embeddings
    print("\n📦 Preparando chunks para embeddings...")
    chunks_para_embedding = []
    
    for chunk in chunks:
        # Formato típico para enviar a API de embeddings
        chunk_data = {
            'id': f"doc_chunk_{chunk.chunk_id}",
            'text': chunk.content,
            'metadata': {
                'page': chunk.metadata.get('page'),
                'section': chunk.metadata.get('section_title'),
                'source': chunk.metadata.get('source_file')
            }
        }
        chunks_para_embedding.append(chunk_data)
    
    print(f"✓ {len(chunks_para_embedding)} chunks listos para embeddings")
    print("\nEjemplo de estructura:")
    print(chunks_para_embedding[0])


def ejemplo_filtrar_chunks():
    """Ejemplo: filtrar chunks por criterios específicos"""
    print("\n" + "="*70)
    print("EJEMPLO 7: FILTRAR CHUNKS")
    print("="*70)
    
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(
        "output_simple/NREL5MW_Reduced/documento_concatenado.md"
    )
    
    # Filtrar solo chunks de la página 1
    chunks_pagina_1 = [
        c for c in chunks 
        if c.metadata.get('page') == 1
    ]
    print(f"\n📄 Chunks de la página 1: {len(chunks_pagina_1)}")
    
    # Filtrar chunks que mencionen "turbine"
    chunks_turbine = [
        c for c in chunks 
        if 'turbine' in c.content.lower()
    ]
    print(f"🔍 Chunks que mencionan 'turbine': {len(chunks_turbine)}")
    
    # Filtrar chunks grandes (>1000 chars)
    chunks_grandes = [
        c for c in chunks 
        if len(c.content) > 1000
    ]
    print(f"📏 Chunks grandes (>1000 chars): {len(chunks_grandes)}")
    
    # Filtrar chunks con títulos de sección
    chunks_con_titulo = [
        c for c in chunks 
        if c.metadata.get('section_title')
    ]
    print(f"📑 Chunks con título de sección: {len(chunks_con_titulo)}")


def menu_principal():
    """Menú interactivo para ejecutar ejemplos"""
    print("\n" + "="*70)
    print("🎓 EJEMPLOS DE USO DEL DOCUMENT CHUNKER")
    print("="*70)
    print("\nSelecciona un ejemplo para ejecutar:")
    print("  1. Uso básico")
    print("  2. Configuración personalizada para RAG conversacional")
    print("  3. Estrategia semántica (máxima coherencia)")
    print("  4. Análisis detallado de chunks")
    print("  5. Función rápida (shortcut)")
    print("  6. Preparar chunks para embeddings")
    print("  7. Filtrar chunks por criterios")
    print("  8. Ejecutar TODOS los ejemplos")
    print("  0. Salir")
    
    ejemplos = {
        '1': ejemplo_basico,
        '2': ejemplo_configuracion_personalizada,
        '3': ejemplo_semantic,
        '4': ejemplo_analisis_chunks,
        '5': ejemplo_funcion_rapida,
        '6': ejemplo_exportar_para_embedding,
        '7': ejemplo_filtrar_chunks
    }
    
    while True:
        try:
            opcion = input("\n👉 Opción: ").strip()
            
            if opcion == '0':
                print("\n¡Hasta luego! 👋")
                break
            elif opcion == '8':
                for func in ejemplos.values():
                    func()
                print("\n✅ Todos los ejemplos ejecutados!")
                break
            elif opcion in ejemplos:
                ejemplos[opcion]()
                input("\nPresiona Enter para continuar...")
            else:
                print("❌ Opción inválida. Intenta de nuevo.")
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Si se pasa un argumento, ejecutar ejemplo específico
    if len(sys.argv) > 1:
        ejemplo_num = sys.argv[1]
        ejemplos = {
            '1': ejemplo_basico,
            '2': ejemplo_configuracion_personalizada,
            '3': ejemplo_semantic,
            '4': ejemplo_analisis_chunks,
            '5': ejemplo_funcion_rapida,
            '6': ejemplo_exportar_para_embedding,
            '7': ejemplo_filtrar_chunks,
            'todos': lambda: [f() for f in [
                ejemplo_basico,
                ejemplo_configuracion_personalizada,
                ejemplo_semantic,
                ejemplo_analisis_chunks,
                ejemplo_funcion_rapida,
                ejemplo_exportar_para_embedding,
                ejemplo_filtrar_chunks
            ]]
        }
        
        if ejemplo_num in ejemplos:
            ejemplos[ejemplo_num]()
        else:
            print(f"❌ Ejemplo '{ejemplo_num}' no encontrado")
            print("Ejemplos disponibles: 1, 2, 3, 4, 5, 6, 7, todos")
    else:
        # Ejecutar menú interactivo
        menu_principal()
