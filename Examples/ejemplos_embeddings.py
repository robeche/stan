"""
Ejemplos de Uso del Módulo de Embeddings
=========================================

Demuestra diferentes formas de usar el generador de embeddings.
"""

from embedding_generator import EmbeddingGenerator
from pathlib import Path


def ejemplo_1_embeddings_basico():
    """Ejemplo básico: Generar embeddings con Nemotron."""
    print("\n" + "="*60)
    print("EJEMPLO 1: Embeddings con NVIDIA Nemotron")
    print("="*60)
    
    # Crear generador con Nemotron (recomendado)
    generator = EmbeddingGenerator(
        model_name="nemotron-v2",
        device="auto",
        batch_size=16
    )
    
    # Generar embedding para un texto
    text = "The NREL 5MW wind turbine is a reference model for offshore wind energy research."
    embedding = generator.generate_embedding(text)
    
    print(f"\nTexto: {text[:80]}...")
    print(f"Dimensión del embedding: {len(embedding)}")
    print(f"Primeros 5 valores: {embedding[:5]}")
    print(f"Norma (si normalizado): {sum(embedding**2)**0.5:.4f}")


def ejemplo_2_procesar_chunks():
    """Ejemplo: Procesar directorio completo de chunks."""
    print("\n" + "="*60)
    print("EJEMPLO 2: Procesar Chunks Completos")
    print("="*60)
    
    # Definir rutas
    chunks_dir = Path("output_simple/NREL5MW_Reduced/chunks_json")
    output_dir = Path("output_rag/embeddings")
    
    if not chunks_dir.exists():
        print(f"⚠ Directorio de chunks no encontrado: {chunks_dir}")
        return
    
    # Crear generador
    generator = EmbeddingGenerator(
        model_name="bge-base",
        device="auto",
        batch_size=8,
        show_progress=True
    )
    
    # Procesar todos los chunks
    metadata = generator.process_chunks_directory(
        chunks_dir=chunks_dir,
        output_dir=output_dir,
        text_field="content",
        save_format="both"  # JSON + numpy
    )
    
    print("\n✓ Proceso completado")
    print(f"  - Chunks procesados: {metadata['num_chunks']}")
    print(f"  - Dimensiones: {metadata['embedding_dimension']}")
    print(f"  - Tiempo: {metadata['generation_time_seconds']:.2f}s")


def ejemplo_3_comparar_modelos():
    """Ejemplo: Comparar diferentes modelos de embeddings."""
    print("\n" + "="*60)
    print("EJEMPLO 3: Comparar Modelos")
    print("="*60)
    
    text = "Wind turbine blade design optimization for offshore applications."
    
    # Modelos a comparar
    modelos = [
        "bge-base",     # BGE (excelente retrieval)
        "minilm",       # Rápido y ligero
        "mpnet",        # Alta calidad
    ]
    
    for model_name in modelos:
        try:
            print(f"\n--- {model_name} ---")
            
            generator = EmbeddingGenerator(
                model_name=model_name,
                device="auto"
            )
            
            embedding = generator.generate_embedding(text)
            
            print(f"Dimensiones: {len(embedding)}")
            print(f"Dispositivo: {generator.get_device()}")
            print(f"Primeros valores: {embedding[:3]}")
            
        except Exception as e:
            print(f"Error: {e}")


def ejemplo_4_embeddings_batch():
    """Ejemplo: Generar embeddings para múltiples textos."""
    print("\n" + "="*60)
    print("EJEMPLO 4: Batch de Embeddings")
    print("="*60)
    
    # Textos de ejemplo
    textos = [
        "Wind turbine aerodynamic performance analysis.",
        "Offshore wind energy cost reduction strategies.",
        "Structural design of wind turbine towers.",
        "Power curve optimization for variable wind conditions.",
        "Environmental impact assessment of wind farms."
    ]
    
    # Crear generador
    generator = EmbeddingGenerator(
        model_name="bge-base",
        batch_size=5
    )
    
    # Generar embeddings en batch
    print(f"\nGenerando embeddings para {len(textos)} textos...")
    embeddings = generator.generate_embeddings_batch(textos)
    
    print(f"\nResultado:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Tipo: {type(embeddings)}")
    
    # Calcular similitudes entre textos
    print(f"\nSimilitud coseno entre textos:")
    for i in range(len(textos)):
        for j in range(i+1, len(textos)):
            similarity = embeddings[i] @ embeddings[j]  # Cosine similarity (ya normalizados)
            print(f"  {i}-{j}: {similarity:.3f}")


def ejemplo_5_listar_modelos():
    """Ejemplo: Listar todos los modelos disponibles."""
    print("\n" + "="*60)
    print("EJEMPLO 5: Modelos Disponibles")
    print("="*60)
    
    modelos = EmbeddingGenerator.list_available_models()
    
    print(f"\nTotal de modelos: {len(modelos)}")
    print("\n" + "-"*60)
    
    for alias, full_name in modelos.items():
        info = EmbeddingGenerator.get_model_info(alias)
        
        print(f"\n{alias}")
        print(f"  Modelo: {full_name}")
        
        if "dimensions" in info:
            print(f"  Dimensiones: {info['dimensions']}")
            print(f"  Mejor para: {info['best_for']}")
            print(f"  Rendimiento: {info['performance']}")
            print(f"  Velocidad: {info['speed']}")


def ejemplo_6_configuracion_avanzada():
    """Ejemplo: Configuración avanzada del generador."""
    print("\n" + "="*60)
    print("EJEMPLO 6: Configuración Avanzada")
    print("="*60)
    
    # Configuración personalizada
    generator = EmbeddingGenerator(
        model_name="bge-base",
        device="cuda",  # Forzar GPU
        batch_size=16,
        normalize_embeddings=True,  # Importante para cosine similarity
        show_progress=True
    )
    
    print(f"\nConfiguración:")
    print(f"  Modelo: {generator.full_model_name}")
    print(f"  Backend: {generator.backend}")
    print(f"  Dispositivo: {generator.get_device()}")
    print(f"  Dimensiones: {generator.get_embedding_dimension()}")
    print(f"  Batch size: {generator.batch_size}")
    print(f"  Normalizado: {generator.normalize_embeddings}")


def menu_interactivo():
    """Menú interactivo para ejecutar ejemplos."""
    ejemplos = {
        "1": ("Embeddings básico con Nemotron", ejemplo_1_embeddings_basico),
        "2": ("Procesar chunks completos", ejemplo_2_procesar_chunks),
        "3": ("Comparar modelos", ejemplo_3_comparar_modelos),
        "4": ("Batch de embeddings", ejemplo_4_embeddings_batch),
        "5": ("Listar modelos disponibles", ejemplo_5_listar_modelos),
        "6": ("Configuración avanzada", ejemplo_6_configuracion_avanzada),
        "7": ("Ejecutar todos los ejemplos", None),
    }
    
    print("\n" + "="*60)
    print("EJEMPLOS DE GENERACIÓN DE EMBEDDINGS")
    print("="*60)
    
    for key, (descripcion, _) in ejemplos.items():
        print(f"{key}. {descripcion}")
    
    print("0. Salir")
    print("="*60)
    
    opcion = input("\nSelecciona un ejemplo: ").strip()
    
    if opcion == "0":
        print("¡Hasta luego!")
        return
    
    if opcion == "7":
        # Ejecutar todos
        for key, (_, func) in ejemplos.items():
            if key != "7" and func:
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
    else:
        print("❌ Opción inválida")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Modo línea de comandos
        ejemplo_num = sys.argv[1]
        
        ejemplos_map = {
            "1": ejemplo_1_embeddings_basico,
            "2": ejemplo_2_procesar_chunks,
            "3": ejemplo_3_comparar_modelos,
            "4": ejemplo_4_embeddings_batch,
            "5": ejemplo_5_listar_modelos,
            "6": ejemplo_6_configuracion_avanzada,
        }
        
        if ejemplo_num in ejemplos_map:
            ejemplos_map[ejemplo_num]()
        else:
            print(f"Uso: python {sys.argv[0]} [1-6]")
    else:
        # Modo interactivo
        menu_interactivo()
