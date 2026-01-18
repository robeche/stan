"""
Script de prueba simple para verificar instalación de embeddings
"""

def verificar_instalacion():
    """Verifica que todas las dependencias estén instaladas correctamente."""
    print("="*60)
    print("VERIFICACIÓN DE INSTALACIÓN")
    print("="*60)
    
    # 1. Verificar torch
    print("\n1. Verificando torch...")
    try:
        import torch
        print(f"   ✓ torch {torch.__version__}")
        print(f"   ✓ CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # 2. Verificar transformers
    print("\n2. Verificando transformers...")
    try:
        import transformers
        print(f"   ✓ transformers {transformers.__version__}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # 3. Verificar sentence-transformers
    print("\n3. Verificando sentence-transformers...")
    try:
        import sentence_transformers
        print(f"   ✓ sentence-transformers {sentence_transformers.__version__}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("\n   Instala con: pip install sentence-transformers")
        return False
    
    # 4. Verificar numpy
    print("\n4. Verificando numpy...")
    try:
        import numpy as np
        print(f"   ✓ numpy {np.__version__}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # 5. Verificar tqdm
    print("\n5. Verificando tqdm...")
    try:
        import tqdm
        print(f"   ✓ tqdm {tqdm.__version__}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ TODAS LAS DEPENDENCIAS INSTALADAS CORRECTAMENTE")
    print("="*60)
    return True


def test_embedding_simple():
    """Prueba simple de generación de embedding."""
    print("\n\n" + "="*60)
    print("TEST DE EMBEDDING SIMPLE")
    print("="*60)
    
    from sentence_transformers import SentenceTransformer
    
    # Usar modelo ligero para la prueba
    print("\nCargando modelo 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Generar embedding
    print("Generando embedding de prueba...")
    text = "This is a test sentence."
    embedding = model.encode(text)
    
    print(f"\n✓ Embedding generado exitosamente!")
    print(f"  Dimensión: {len(embedding)}")
    print(f"  Primeros 5 valores: {embedding[:5]}")
    print(f"  Tipo: {type(embedding)}")
    
    return True


if __name__ == "__main__":
    # Verificar instalación
    if verificar_instalacion():
        # Si todo está bien, probar embedding
        try:
            test_embedding_simple()
        except Exception as e:
            print(f"\n❌ Error al generar embedding: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n❌ Por favor, instala las dependencias faltantes antes de continuar.")
