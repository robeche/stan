# 🔒 Guía de Seguridad del Repositorio

## Archivos Sensibles Protegidos

Este repositorio está configurado para **NO incluir**:

### 1. Tokens y Credenciales
- ✅ Archivos `.env` con API keys
- ✅ Configuraciones locales con credenciales
- ✅ Archivos `secrets.json` o similares

### 2. Modelos de Machine Learning
- ✅ Archivos `.pt`, `.pth`, `.pkl`, `.h5`, `.bin`
- ✅ Modelos de PyTorch, TensorFlow, ONNX
- ✅ Archivos `.safetensors`

### 3. Bases de Datos
- ✅ Archivos SQLite (`.sqlite3`, `.db`)
- ✅ Bases de datos vectoriales (ChromaDB, FAISS)
- ✅ Archivos de journaling de BD

### 4. Outputs y Datos Generados
- ✅ Directorios `output_rag/`, `output_simple/`
- ✅ Embeddings pre-computados
- ✅ Archivos procesados

---

## ⚙️ Configuración Inicial

### 1. Crear archivo `.env` local

```bash
cp .env.example .env
```

Edita el archivo `.env` y añade tus credenciales:

```env
# OpenAI API Key (para embeddings)
OPENAI_API_KEY=sk-proj-TU-API-KEY-REAL-AQUI

# Otras API keys si las usas
NGC_API_KEY=tu-nvidia-key
```

### 2. Configurar Django (WebApp)

Edita `WebApp/rag_project/settings.py`:

```python
# Cambia la SECRET_KEY en producción
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'tu-key-de-desarrollo')

# Asegúrate de usar variables de entorno para APIs
NVIDIA_API_KEY = os.getenv('NGC_API_KEY')
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
cd WebApp
pip install -r requirements.txt
```

---

## 🚨 Antes de Hacer Push

### Verificar archivos a subir

```bash
git status
```

### Verificar que NO se suban archivos sensibles

```bash
# Ver qué archivos están siendo rastreados
git ls-files

# Ver archivos ignorados
git status --ignored
```

### ⚠️ Si accidentalmente se subió información sensible:

1. **Eliminar del historial:**
```bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch ARCHIVO_SENSIBLE" \
  --prune-empty --tag-name-filter cat -- --all
```

2. **Rotar credenciales:** Cambia inmediatamente cualquier API key o token expuesto.

---

## 📦 Archivos que SÍ se deben incluir

✅ Código fuente (`.py`)
✅ Archivos de configuración de ejemplo (`.env.example`)
✅ READMEs y documentación (`.md`)
✅ Requirements (`.txt`)
✅ Notebooks de ejemplo (`.ipynb`) sin outputs sensibles
✅ Archivos de configuración de estructura (`memoria/*.tex`)

---

## 🔍 Verificación de Seguridad

### Comando para buscar posibles tokens hardcodeados:

```bash
# Buscar patrones de API keys
grep -r "sk-" --include="*.py" .
grep -r "api_key.*=" --include="*.py" .

# Buscar en settings.py
grep -i "secret\|password\|key" WebApp/rag_project/settings.py
```

### Auditar archivos grandes antes de push:

```bash
# Ver tamaño de archivos
find . -type f -size +10M -not -path "./venv/*" -not -path "./.git/*"
```

---

## 📝 Checklist Pre-Push

- [ ] Verificar que `.env` no está en el staging area
- [ ] Confirmar que no hay modelos (`.pt`, `.pkl`) siendo añadidos
- [ ] Revisar que bases de datos (`.sqlite3`, `chroma_db/`) están ignoradas
- [ ] Verificar que directorios de output están ignorados
- [ ] Confirmar que `venv/` no está siendo rastreado
- [ ] Revisar commits con `git diff --cached` antes de hacer push
- [ ] Asegurar que tokens en código están usando variables de entorno

---

## 🛠️ Inicializar Repositorio

```bash
# Inicializar Git
git init

# Añadir archivos (respetando .gitignore)
git add .

# Ver qué se va a commitear
git status

# Primer commit
git commit -m "Initial commit: RAG project structure"

# Conectar con repositorio remoto
git remote add origin https://github.com/TU-USUARIO/TU-REPO.git

# Push
git branch -M main
git push -u origin main
```

---

## 📚 Recursos Adicionales

- [GitHub - Eliminar datos sensibles](https://docs.github.com/es/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [Git Secrets](https://github.com/awslabs/git-secrets) - Herramienta para prevenir commits de secretos
- [GitGuardian](https://www.gitguardian.com/) - Escaneo automático de secretos

---

## ⚡ Solución Rápida si se Expuso un Token

1. **Rotar inmediatamente:**
   - OpenAI: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Otros servicios: Revisar sus respectivos paneles

2. **Limpiar historial (si es necesario):**
```bash
# Usar git-filter-repo (recomendado)
pip install git-filter-repo
git filter-repo --invert-paths --path ARCHIVO_CON_TOKEN
```

3. **Force push (PRECAUCIÓN):**
```bash
git push origin --force --all
```
