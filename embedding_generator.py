"""
Módulo de Generación de Embeddings
====================================

Genera embeddings vectoriales para chunks de documentos usando diferentes modelos.
Soporta modelos locales (Sentence Transformers, Nemotron) y APIs externas (OpenAI).

Autor: Sistema RAG
Fecha: 2026-01-02
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm import tqdm
import time


class EmbeddingGenerator:
    """
    Generador de embeddings vectoriales para documentos.
    
    Soporta múltiples backends:
    - Sentence Transformers (HuggingFace)
    - NVIDIA Nemotron
    - OpenAI API
    """
    
    SUPPORTED_MODELS = {
        # NVIDIA Nemotron (RECOMENDADO para RAG)
        "nemotron-v2": "nvidia/NV-Embed-v2",
        "nemotron-v1": "nvidia/NV-Embed-v1",
        
        # Sentence Transformers - Balance calidad/velocidad
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "mpnet": "sentence-transformers/all-mpnet-base-v2",
        
        # BGE Models - Excelente para retrieval
        "bge-m3": "BAAI/bge-m3",
        "bge-small": "BAAI/bge-small-en-v1.5",
        "bge-base": "BAAI/bge-base-en-v1.5",
        "bge-large": "BAAI/bge-large-en-v1.5",
        
        # Multilingüe
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        
        # OpenAI
        "openai-small": "text-embedding-3-small",
        "openai-large": "text-embedding-3-large",
    }
    
    def __init__(
        self,
        model_name: str = "bge-m3",
        device: str = "auto",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Inicializa el generador de embeddings.
        
        Args:
            model_name: Nombre del modelo (usar keys de SUPPORTED_MODELS)
            device: Dispositivo ('cuda', 'cpu', 'auto')
            batch_size: Tamaño del batch para procesamiento
            normalize_embeddings: Si normalizar embeddings (recomendado para cosine similarity)
            show_progress: Mostrar barra de progreso
            api_key: API key para modelos externos (OpenAI)
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress = show_progress
        self.api_key = api_key
        
        # Obtener nombre completo del modelo
        if model_name in self.SUPPORTED_MODELS:
            self.full_model_name = self.SUPPORTED_MODELS[model_name]
        else:
            self.full_model_name = model_name
        
        # Determinar backend
        self.backend = self._determine_backend()
        
        # Cargar modelo
        self.model = self._load_model()
        
        print(f"✓ Modelo cargado: {self.full_model_name}")
        print(f"  Backend: {self.backend}")
        print(f"  Dimensiones: {self.get_embedding_dimension()}")
        print(f"  Dispositivo: {self.get_device()}")
    
    def _determine_backend(self) -> str:
        """Determina qué backend usar según el modelo."""
        if "openai" in self.model_name or "text-embedding" in self.full_model_name:
            return "openai"
        else:
            return "sentence-transformers"
    
    def _load_model(self):
        """Carga el modelo según el backend."""
        if self.backend == "openai":
            return self._load_openai_model()
        else:
            return self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        """Carga modelo de Sentence Transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Determinar dispositivo
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Determinar si necesita trust_remote_code (para Nemotron y otros)
            trust_remote_code = "nvidia" in self.full_model_name.lower()
            
            # Cargar modelo
            model = SentenceTransformer(
                self.full_model_name,
                device=device,
                trust_remote_code=trust_remote_code
            )
            
            return model
            
        except ImportError:
            raise ImportError(
                "Se requiere 'sentence-transformers'. "
                "Instala con: pip install sentence-transformers"
            )
    
    def _load_openai_model(self):
        """Configura cliente de OpenAI."""
        try:
            from openai import OpenAI
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Se requiere API key de OpenAI. "
                    "Proporciona 'api_key' o establece OPENAI_API_KEY"
                )
            
            return OpenAI(api_key=api_key)
            
        except ImportError:
            raise ImportError(
                "Se requiere 'openai'. "
                "Instala con: pip install openai"
            )
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Genera embedding para un texto individual.
        
        Args:
            text: Texto para generar embedding
            
        Returns:
            Vector de embedding como numpy array
        """
        if self.backend == "openai":
            return self._generate_openai_embedding(text)
        else:
            return self._generate_sentence_transformer_embedding(text)
    
    def _generate_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Genera embedding con Sentence Transformers."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False
        )
        return embedding
    
    def _generate_openai_embedding(self, text: str) -> np.ndarray:
        """Genera embedding con OpenAI."""
        response = self.model.embeddings.create(
            model=self.full_model_name,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Genera embeddings para múltiples textos.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Array de embeddings (n_texts, embedding_dim)
        """
        if self.backend == "openai":
            return self._generate_openai_embeddings_batch(texts)
        else:
            return self._generate_sentence_transformer_embeddings_batch(texts)
    
    def _generate_sentence_transformer_embeddings_batch(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """Genera embeddings batch con Sentence Transformers."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress
        )
        return embeddings
    
    def _generate_openai_embeddings_batch(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """Genera embeddings batch con OpenAI."""
        all_embeddings = []
        
        # Procesar en batches
        iterator = range(0, len(texts), self.batch_size)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Generando embeddings")
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            
            response = self.model.embeddings.create(
                model=self.full_model_name,
                input=batch
            )
            
            batch_embeddings = [
                np.array(item.embedding) 
                for item in response.data
            ]
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting
            time.sleep(0.1)
        
        return np.array(all_embeddings)
    
    def process_chunks_directory(
        self,
        chunks_dir: Union[str, Path],
        output_dir: Union[str, Path],
        text_field: str = "content",
        save_format: str = "json"
    ) -> Dict:
        """
        Procesa un directorio de chunks y genera embeddings.
        
        Args:
            chunks_dir: Directorio con chunks en JSON
            output_dir: Directorio para guardar embeddings
            text_field: Campo del JSON que contiene el texto
            save_format: Formato de salida ('json', 'npy', 'both')
            
        Returns:
            Diccionario con estadísticas del proceso
        """
        chunks_dir = Path(chunks_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar todos los chunks
        print("Cargando chunks...")
        chunk_files = sorted(chunks_dir.glob("chunk_*.json"))
        
        if not chunk_files:
            raise ValueError(f"No se encontraron chunks en {chunks_dir}")
        
        chunks_data = []
        texts = []
        
        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                chunks_data.append(chunk_data)
                texts.append(chunk_data.get(text_field, ""))
        
        print(f"✓ Cargados {len(chunks_data)} chunks")
        
        # Generar embeddings
        print("\nGenerando embeddings...")
        start_time = time.time()
        embeddings = self.generate_embeddings_batch(texts)
        elapsed_time = time.time() - start_time
        
        print(f"✓ Embeddings generados en {elapsed_time:.2f}s")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Tiempo por chunk: {elapsed_time/len(chunks_data):.3f}s")
        
        # Guardar embeddings
        print("\nGuardando embeddings...")
        self._save_embeddings(
            chunks_data,
            embeddings,
            output_dir,
            save_format
        )
        
        # Guardar metadata
        metadata = {
            "model": self.full_model_name,
            "model_alias": self.model_name,
            "embedding_dimension": int(embeddings.shape[1]),
            "num_chunks": len(chunks_data),
            "normalized": self.normalize_embeddings,
            "generation_time_seconds": elapsed_time,
            "backend": self.backend,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = output_dir / "embeddings_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Embeddings guardados en: {output_dir}")
        print(f"✓ Metadata guardada en: {metadata_path}")
        
        return metadata
    
    def _save_embeddings(
        self,
        chunks_data: List[Dict],
        embeddings: np.ndarray,
        output_dir: Path,
        save_format: str
    ):
        """Guarda embeddings en el formato especificado."""
        
        if save_format in ["json", "both"]:
            # Guardar como JSON con chunks
            for i, (chunk_data, embedding) in enumerate(zip(chunks_data, embeddings)):
                chunk_with_embedding = chunk_data.copy()
                chunk_with_embedding["embedding"] = embedding.tolist()
                chunk_with_embedding["embedding_model"] = self.full_model_name
                
                output_file = output_dir / f"chunk_{i:04d}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_with_embedding, f, indent=2, ensure_ascii=False)
        
        if save_format in ["npy", "both"]:
            # Guardar embeddings como numpy array
            npy_file = output_dir / "embeddings.npy"
            np.save(npy_file, embeddings)
            
            # Guardar índice de chunks
            index_data = [
                {
                    "chunk_id": chunk.get("chunk_id", i),
                    "chunk_file": f"chunk_{i:04d}.json"
                }
                for i, chunk in enumerate(chunks_data)
            ]
            
            index_file = output_dir / "embeddings_index.json"
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
    
    def get_embedding_dimension(self) -> int:
        """Obtiene la dimensión de los embeddings."""
        if self.backend == "openai":
            # Dimensiones conocidas de OpenAI
            if "small" in self.full_model_name:
                return 1536
            elif "large" in self.full_model_name:
                return 3072
            else:
                return 1536
        else:
            # Generar embedding de prueba
            test_embedding = self.model.encode(
                "test",
                normalize_embeddings=self.normalize_embeddings
            )
            return len(test_embedding)
    
    def get_device(self) -> str:
        """Obtiene el dispositivo en uso."""
        if self.backend == "openai":
            return "api"
        else:
            return str(self.model.device)
    
    @staticmethod
    def list_available_models() -> Dict[str, str]:
        """Lista todos los modelos soportados."""
        return EmbeddingGenerator.SUPPORTED_MODELS.copy()
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict:
        """Obtiene información sobre un modelo específico."""
        models_info = {
            "nemotron-v2": {
                "name": "nvidia/NV-Embed-v2",
                "dimensions": 4096,
                "max_tokens": 32768,
                "best_for": "RAG, documentos técnicos/científicos",
                "performance": "⭐⭐⭐⭐⭐",
                "speed": "⭐⭐⭐"
            },
            "nemotron-v1": {
                "name": "nvidia/NV-Embed-v1",
                "dimensions": 4096,
                "max_tokens": 32768,
                "best_for": "RAG general",
                "performance": "⭐⭐⭐⭐",
                "speed": "⭐⭐⭐"
            },
            "minilm": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "max_tokens": 512,
                "best_for": "Balance velocidad/calidad",
                "performance": "⭐⭐⭐",
                "speed": "⭐⭐⭐⭐⭐"
            },
            "mpnet": {
                "name": "sentence-transformers/all-mpnet-base-v2",
                "dimensions": 768,
                "max_tokens": 512,
                "best_for": "Alta calidad general",
                "performance": "⭐⭐⭐⭐",
                "speed": "⭐⭐⭐⭐"
            },
            "bge-m3": {
                "name": "BAAI/bge-m3",
                "dimensions": 1024,
                "max_tokens": 8192,
                "best_for": "Multilingüe, última generación",
                "performance": "⭐⭐⭐⭐⭐",
                "speed": "⭐⭐⭐⭐"
            },
            "bge-small": {
                "name": "BAAI/bge-small-en-v1.5",
                "dimensions": 384,
                "max_tokens": 512,
                "best_for": "Retrieval rápido",
                "performance": "⭐⭐⭐⭐",
                "speed": "⭐⭐⭐⭐⭐"
            },
            "bge-base": {
                "name": "BAAI/bge-base-en-v1.5",
                "dimensions": 768,
                "max_tokens": 512,
                "best_for": "Retrieval de alta calidad",
                "performance": "⭐⭐⭐⭐⭐",
                "speed": "⭐⭐⭐⭐"
            },
            "bge-large": {
                "name": "BAAI/bge-large-en-v1.5",
                "dimensions": 1024,
                "max_tokens": 512,
                "best_for": "Máxima precisión en retrieval",
                "performance": "⭐⭐⭐⭐⭐",
                "speed": "⭐⭐⭐"
            },
        }
        
        return models_info.get(model_name, {"name": model_name})


def main():
    """Función principal para uso desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Genera embeddings para chunks de documentos"
    )
    parser.add_argument(
        "chunks_dir",
        type=str,
        help="Directorio con chunks en formato JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directorio de salida para embeddings (default: chunks_dir/embeddings)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nemotron-v2",
        help=f"Modelo a usar. Opciones: {', '.join(EmbeddingGenerator.SUPPORTED_MODELS.keys())}"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño del batch para procesamiento"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Dispositivo para el modelo"
    )
    parser.add_argument(
        "--save-format",
        type=str,
        default="both",
        choices=["json", "npy", "both"],
        help="Formato de salida de embeddings"
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="content",
        help="Campo del JSON que contiene el texto"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Listar modelos disponibles y salir"
    )
    
    args = parser.parse_args()
    
    # Listar modelos si se solicita
    if args.list_models:
        print("\n=== MODELOS DISPONIBLES ===\n")
        for alias, full_name in EmbeddingGenerator.list_available_models().items():
            info = EmbeddingGenerator.get_model_info(alias)
            print(f"{alias:20s} -> {full_name}")
            if "dimensions" in info:
                print(f"  Dimensiones: {info['dimensions']}")
                print(f"  Mejor para: {info['best_for']}")
                print(f"  Rendimiento: {info['performance']}")
                print(f"  Velocidad: {info['speed']}")
            print()
        return
    
    # Determinar directorio de salida
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.chunks_dir, "..", "embeddings")
    
    print("="*60)
    print("GENERADOR DE EMBEDDINGS")
    print("="*60)
    print(f"Chunks: {args.chunks_dir}")
    print(f"Output: {output_dir}")
    print(f"Modelo: {args.model}")
    print("="*60)
    print()
    
    # Crear generador
    generator = EmbeddingGenerator(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress=True
    )
    
    # Procesar chunks
    metadata = generator.process_chunks_directory(
        chunks_dir=args.chunks_dir,
        output_dir=output_dir,
        text_field=args.text_field,
        save_format=args.save_format
    )
    
    print("\n" + "="*60)
    print("PROCESO COMPLETADO")
    print("="*60)
    print(f"✓ Chunks procesados: {metadata['num_chunks']}")
    print(f"✓ Dimensiones: {metadata['embedding_dimension']}")
    print(f"✓ Tiempo total: {metadata['generation_time_seconds']:.2f}s")
    print(f"✓ Embeddings guardados en: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
