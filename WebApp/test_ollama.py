"""
Test Ollama connection and list available models
"""

import requests
import json

OLLAMA_URL = "http://localhost:11434"

print("=" * 80)
print("OLLAMA CONNECTION TEST")
print("=" * 80)

# Test 1: Check if Ollama is running
print(f"\n1. Verificando conexión a {OLLAMA_URL}...")
try:
    response = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
    if response.status_code == 200:
        print(f"   ✓ Ollama está corriendo")
        print(f"   Versión: {response.json().get('version', 'Unknown')}")
    else:
        print(f"   ✗ Error: {response.status_code}")
except requests.exceptions.ConnectionError:
    print(f"   ✗ No se pudo conectar. Asegúrate de que Docker esté corriendo.")
    print(f"   Comando: docker ps | grep ollama")
    exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 2: List available models
print(f"\n2. Listando modelos disponibles...")
try:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    if response.status_code == 200:
        data = response.json()
        models = data.get('models', [])
        
        if models:
            print(f"   ✓ Encontrados {len(models)} modelos:")
            for model in models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"   - {name} ({size:.2f} GB)")
        else:
            print(f"   ⚠ No hay modelos instalados")
            print(f"\n   Para instalar un modelo, ejecuta:")
            print(f"   docker exec -it <container_name> ollama pull llama3.2")
            print(f"   docker exec -it <container_name> ollama pull mistral")
    else:
        print(f"   ✗ Error al listar modelos: {response.status_code}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Simple generation test
print(f"\n3. Probando generación de texto...")
try:
    # Get first available model
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    models = response.json().get('models', [])
    
    if models:
        test_model = models[0].get('name')
        print(f"   Usando modelo: {test_model}")
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": test_model,
                "prompt": "Di 'hola' en español",
                "stream": False,
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            print(f"   ✓ Respuesta: {answer[:100]}")
            print(f"   Tiempo: {result.get('total_duration', 0) / 1e9:.2f}s")
        else:
            print(f"   ✗ Error en generación: {response.status_code}")
    else:
        print(f"   ⚠ No hay modelos para probar")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("RESUMEN")
print("=" * 80)

# Get recommended model
try:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    models = response.json().get('models', [])
    
    if models:
        print(f"\n✓ Ollama está listo para usar")
        print(f"\n📝 Modelos disponibles:")
        for model in models:
            name = model.get('name', 'Unknown')
            print(f"   - {name}")
        
        print(f"\n⚙️ Para usar en el chatbot, actualiza settings.py:")
        print(f"   OLLAMA_CONFIG = {{")
        print(f"       'MODEL': '{models[0].get('name')}',  # Cambia al que prefieras")
        print(f"   }}")
        
        print(f"\n🚀 El chatbot está listo para generar respuestas con IA!")
    else:
        print(f"\n⚠ Instala un modelo primero:")
        print(f"   docker exec -it <container_name> ollama pull llama3.2")
        
except Exception as e:
    print(f"\n✗ No se pudo verificar los modelos: {e}")
