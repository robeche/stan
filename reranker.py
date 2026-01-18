"""
Módulo de Reranking para Sistema RAG
=====================================

Reordena resultados de búsqueda vectorial usando modelos cross-encoder
para mejorar la precisión de retrieval.

Autor: Sistema RAG
Fecha: 2026-01-02
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class RerankResult:
    """Resultado de reranking con documento y puntaje."""
    id: str
    document: str
    metadata: Dict
    original_score: float
    rerank_score: float
    rank_change: int


class Reranker:
    """
    Reranker basado en modelos cross-encoder.
    
    Mejora la precisión de búsquedas vectoriales al analizar
    la relación entre query y documento de forma conjunta.
    """
    
    SUPPORTED_MODELS = {
        # BGE Rerankers (BAAI - Recomendados)
        "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
        "bge-reranker-base": "BAAI/bge-reranker-base",
        "bge-reranker-large": "BAAI/bge-reranker-large",
        
        # MS MARCO (Rápidos)
        "ms-marco-mini": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-small": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        
        # Multilingües
        "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    }
    
    def __init__(
        self,
        model_name: str = "bge-reranker-v2-m3",
        device: str = "auto",
        batch_size: int = 32,
        show_progress: bool = False
    ):
        """
        Inicializa el reranker.
        
        Args:
            model_name: Nombre del modelo (usar keys de SUPPORTED_MODELS)
            device: Dispositivo ('cuda', 'cpu', 'auto')
            batch_size: Tamaño del batch para procesamiento
            show_progress: Mostrar barra de progreso
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # Obtener nombre completo del modelo
        if model_name in self.SUPPORTED_MODELS:
            self.full_model_name = self.SUPPORTED_MODELS[model_name]
        else:
            self.full_model_name = model_name
        
        # Cargar modelo
        self.model = self._load_model()
        
        print(f"✓ Reranker cargado: {self.full_model_name}")
        print(f"  Dispositivo: {self.get_device()}")
    
    def _load_model(self):
        """Carga el modelo de reranking."""
        try:
            from sentence_transformers import CrossEncoder
            import torch
            
            # Determinar dispositivo
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Cargar modelo
            model = CrossEncoder(
                self.full_model_name,
                device=device,
                max_length=512
            )
            
            return model
            
        except ImportError:
            raise ImportError(
                "Se requiere 'sentence-transformers'. "
                "Instala con: pip install sentence-transformers"
            )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Reordena documentos según relevancia con la query.
        
        Args:
            query: Texto de consulta
            documents: Lista de documentos a reordenar
            top_k: Número de documentos a retornar (None = todos)
            
        Returns:
            Lista de tuplas (índice_original, puntaje) ordenada por puntaje
        """
        # Crear pares (query, documento)
        pairs = [[query, doc] for doc in documents]
        
        # Obtener puntajes
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress
        )
        
        # Crear lista de (índice, puntaje) y ordenar
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Limitar a top_k si se especifica
        if top_k is not None:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores
    
    def rerank_results(
        self,
        query: str,
        search_results: Dict,
        top_k: Optional[int] = None,
        return_full_results: bool = True
    ) -> Union[List[RerankResult], Dict]:
        """
        Reordena resultados de búsqueda vectorial.
        
        Args:
            query: Texto de consulta
            search_results: Resultados de VectorStore.query()
            top_k: Número de resultados a retornar
            return_full_results: Si retornar objetos RerankResult completos
            
        Returns:
            Lista de RerankResult o diccionario con resultados formateados
        """
        # Extraer documentos
        results = search_results.get('results', [])
        documents = [r['document'] for r in results]
        
        if not documents:
            return [] if return_full_results else {'results': []}
        
        # Reordenar
        reranked_indices = self.rerank(query, documents, top_k=None)
        
        # Crear resultados reordenados
        reranked_results = []
        
        for new_rank, (original_idx, rerank_score) in enumerate(reranked_indices):
            result = results[original_idx]
            
            rank_change = original_idx - new_rank
            
            rerank_result = RerankResult(
                id=result['id'],
                document=result['document'],
                metadata=result['metadata'],
                original_score=result.get('similarity', result.get('distance', 0)),
                rerank_score=float(rerank_score),
                rank_change=rank_change
            )
            
            reranked_results.append(rerank_result)
        
        # Limitar a top_k si se especifica
        if top_k is not None:
            reranked_results = reranked_results[:top_k]
        
        if return_full_results:
            return reranked_results
        
        # Formato compatible con VectorStore
        return {
            'query': query,
            'n_results': len(reranked_results),
            'results': [
                {
                    'id': r.id,
                    'document': r.document,
                    'metadata': r.metadata,
                    'rerank_score': r.rerank_score,
                    'original_score': r.original_score,
                    'rank_change': r.rank_change
                }
                for r in reranked_results
            ]
        }
    
    def score_pairs(
        self,
        query: str,
        documents: List[str]
    ) -> np.ndarray:
        """
        Calcula puntajes para pares query-documento.
        
        Args:
            query: Texto de consulta
            documents: Lista de documentos
            
        Returns:
            Array de puntajes
        """
        pairs = [[query, doc] for doc in documents]
        
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress
        )
        
        return scores
    
    def get_device(self) -> str:
        """Obtiene el dispositivo en uso."""
        import torch
        if hasattr(self.model, 'model'):
            return str(self.model.model.device)
        return "unknown"
    
    @staticmethod
    def list_available_models() -> Dict[str, str]:
        """Lista todos los modelos soportados."""
        return Reranker.SUPPORTED_MODELS.copy()
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict:
        """Obtiene información sobre un modelo específico."""
        models_info = {
            "bge-reranker-v2-m3": {
                "name": "BAAI/bge-reranker-v2-m3",
                "max_length": 8192,
                "languages": "Multilingüe",
                "best_for": "RAG, última generación",
                "performance": "⭐⭐⭐⭐⭐",
                "speed": "⭐⭐⭐⭐"
            },
            "bge-reranker-base": {
                "name": "BAAI/bge-reranker-base",
                "max_length": 512,
                "languages": "Inglés",
                "best_for": "Balance rendimiento/velocidad",
                "performance": "⭐⭐⭐⭐",
                "speed": "⭐⭐⭐⭐"
            },
            "bge-reranker-large": {
                "name": "BAAI/bge-reranker-large",
                "max_length": 512,
                "languages": "Inglés",
                "best_for": "Máxima precisión",
                "performance": "⭐⭐⭐⭐⭐",
                "speed": "⭐⭐⭐"
            },
            "ms-marco-mini": {
                "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "max_length": 512,
                "languages": "Inglés",
                "best_for": "Velocidad máxima",
                "performance": "⭐⭐⭐",
                "speed": "⭐⭐⭐⭐⭐"
            },
            "ms-marco-small": {
                "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                "max_length": 512,
                "languages": "Inglés",
                "best_for": "Búsquedas web",
                "performance": "⭐⭐⭐⭐",
                "speed": "⭐⭐⭐⭐"
            },
        }
        
        return models_info.get(model_name, {"name": model_name})


def main():
    """Función principal para uso desde línea de comandos."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Reranking de resultados de búsqueda"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Texto de consulta"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Archivo JSON con resultados de búsqueda"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge-reranker-v2-m3",
        help="Modelo de reranking a usar"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Número de resultados a retornar"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Dispositivo para el modelo"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Listar modelos disponibles y salir"
    )
    
    args = parser.parse_args()
    
    # Listar modelos si se solicita
    if args.list_models:
        print("\n=== MODELOS DE RERANKING DISPONIBLES ===\n")
        for alias, full_name in Reranker.list_available_models().items():
            info = Reranker.get_model_info(alias)
            print(f"{alias:25s} -> {full_name}")
            if "max_length" in info:
                print(f"  Max length: {info['max_length']}")
                print(f"  Idiomas: {info['languages']}")
                print(f"  Mejor para: {info['best_for']}")
                print(f"  Rendimiento: {info['performance']}")
                print(f"  Velocidad: {info['speed']}")
            print()
        return
    
    print("="*60)
    print("RERANKING DE RESULTADOS")
    print("="*60)
    print(f"Query: {args.query}")
    print(f"Modelo: {args.model}")
    print(f"Top-K: {args.top_k}")
    print("="*60)
    print()
    
    # Cargar resultados
    with open(args.results_file, 'r', encoding='utf-8') as f:
        search_results = json.load(f)
    
    # Crear reranker
    reranker = Reranker(
        model_name=args.model,
        device=args.device,
        show_progress=True
    )
    
    # Reordenar
    reranked = reranker.rerank_results(
        query=args.query,
        search_results=search_results,
        top_k=args.top_k,
        return_full_results=False
    )
    
    # Mostrar resultados
    print("\n🎯 RESULTADOS REORDENADOS:\n")
    
    for i, result in enumerate(reranked['results'], 1):
        change_symbol = "↑" if result['rank_change'] > 0 else "↓" if result['rank_change'] < 0 else "="
        
        print(f"{i}. {result['id']}")
        print(f"   Rerank Score: {result['rerank_score']:.4f}")
        print(f"   Original Score: {result['original_score']:.4f}")
        print(f"   Rank Change: {change_symbol} {abs(result['rank_change'])}")
        print(f"   Document: {result['document'][:100]}...")
        print()


if __name__ == "__main__":
    main()
