import time
import random
import logging
from typing import List, Generator, Optional

logger = logging.getLogger(__name__)

# Intentamos importar PyTorch y Hugging Face Transformers para soportar inferencia local real
TRANSFORMERS_AVAILABLE = False
torch = None
AutoTokenizer = None
AutoModelForCausalLM = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class InferenceEngineError(Exception):
    """Excepcion base para fallos en el motor de inferencia."""
    pass


class InferenceEngine:
    """
    Motor de inferencia flexible que maneja la generacion de texto.
    
    Soporta inferencia real mediante modelos CausalLM locales (Transformers)
    y un motor de simulacion semantica robusto en caso de estar offline o sin GPU,
    garantizando que el sistema sea testeable y funcional en cualquier entorno.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        use_local_model: bool = True
    ) -> None:
        """
        Args:
            model_name: Nombre del modelo en Hugging Face para inferencia real.
            use_local_model: Si es True, intentara cargar el modelo localmente en GPU/CPU.
        """
        self.model_name = model_name
        self.use_local = use_local_model and TRANSFORMERS_AVAILABLE
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        
        if self.use_local:
            try:
                # Seleccion automatica del dispositivo de computo acelerado
                if torch.backends.mps.is_available():
                    self.device = "mps"
                elif torch.cuda.is_available():
                    self.device = "cuda"
                
                logger.info(f"Cargando modelo local '{model_name}' en '{self.device}'...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Modelo local cargado exitosamente.")
            except Exception as e:
                logger.warning(
                    f"No se pudo cargar el modelo '{model_name}'. "
                    f"Cambiando a modo de simulacion offline. Motivo: {str(e)}"
                )
                self.use_local = False

    def generate_batch(self, prompts: List[str], max_tokens: int = 100) -> List[str]:
        """
        Procesa un lote de prompts simultaneamente utilizando el modelo disponible.
        """
        if not prompts:
            return []
            
        if self.use_local and self.model and self.tokenizer:
            try:
                # La tokenizacion por lotes requiere configurar el pad_token
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # Decodificamos solo la parte generada de cada secuencia
                decoded = []
                for i, out in enumerate(outputs):
                    input_len = inputs["input_ids"][i].shape[0]
                    gen_tokens = out[input_len:]
                    decoded.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))
                return decoded
            except Exception as e:
                logger.error(f"Error en inferencia por lotes local: {str(e)}. Usando fallback offline.")
                return [self._simulate_generation(p, max_tokens) for p in prompts]
        else:
            # Simulacion multi-hilo o secuencial offline
            return [self._simulate_generation(p, max_tokens) for p in prompts]

    def generate_stream(self, prompt: str, max_tokens: int = 100) -> Generator[str, None, None]:
        """
        Genera tokens de forma secuencial emitiendo fragmentos (streaming).
        """
        if self.use_local and self.model and self.tokenizer:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Para streaming real en transformers podemos usar TextIteratorStreamer,
                # pero para mantener el codigo auto-contenido y compatible con modelos tiny,
                # simulamos el streaming token a token llamando progresivamente o decodificando.
                # Como alternativa robusta y rapida, yieldamos palabras del output para simular streaming.
                full_text = self.generate_batch([prompt], max_tokens=max_tokens)[0]
                words = full_text.split(" ")
                for word in words:
                    yield word + " "
                    time.sleep(0.03)  # Simulacion de latencia de red/procesamiento de token
            except Exception as e:
                logger.error(f"Error en streaming local: {str(e)}. Usando fallback offline.")
                for chunk in self._simulate_stream(prompt, max_tokens):
                    yield chunk
        else:
            for chunk in self._simulate_stream(prompt, max_tokens):
                yield chunk

    def _simulate_generation(self, prompt: str, max_tokens: int) -> str:
        """
        Genera una respuesta simulada coherente basada en palabras clave del prompt.
        """
        prompt_lower = prompt.lower()
        
        # Base de conocimiento simulada para construir respuestas realistas
        templates = {
            "rag": (
                "La arquitectura RAG (Generacion Aumentada por Recuperación) combina la precision "
                "de busquedas clasicas BM25 y semanticas vectoriales HNSW. El flujo primero recupera "
                "documentos relevantes usando NanoVectorDB y despues inyecta ese contexto filtrado "
                "por Cross-Encoder en la ventana de atencion del LLM, reduciendo alucinaciones."
            ),
            "vector": (
                "Las bases de datos vectoriales indexan embeddings de alta dimension en grafos HNSW. "
                "Esto permite realizar consultas de similitud de coseno en milisegundos, localizando "
                "documentos por su afinidad conceptual en vez de coincidencias de caracteres literales."
            ),
            "token": (
                "El algoritmo BPE (Byte Pair Encoding) segmenta el texto plano en unidades sub-palabra. "
                "Evita el problema de palabras fuera de vocabulario codificando secuencias frecuentes "
                "de bytes y asignando a cada token un identificador entero unico para el modelo."
            ),
            "agent": (
                "Los agentes autonomos operan bajo bucles de razonamiento ReAct (Thought-Action-Observation). "
                "Son capaces de planificar tareas complejas, evaluar resultados de ejecucion en sandboxes "
                "seguros, y persistir memoria a corto y largo plazo conectandose a bases de datos vectoriales."
            )
        }
        
        # Buscamos coincidencias semanticas
        response = None
        for key, text in templates.items():
            if key in prompt_lower:
                response = text
                break
                
        if not response:
            response = (
                "Procesando prompt en el servidor de inferencia. "
                "Esta es una respuesta simulada de alta fidelidad que contiene informacion tecnica "
                "sobre arquitectura de infraestructura de Inteligencia Artificial modular y escalable."
            )
            
        # Truncamos segun max_tokens estimado (aprox. 1 palabra = 1.3 tokens)
        words = response.split(" ")
        num_words = int(max_tokens / 1.3)
        return " ".join(words[:num_words])

    def _simulate_stream(self, prompt: str, max_tokens: int) -> Generator[str, None, None]:
        """
        Generador de streaming simulado para responder palabra por palabra.
        """
        text = self._simulate_generation(prompt, max_tokens)
        words = text.split(" ")
        for word in words:
            # Yieldamos la palabra con un espacio al final
            yield word + " "
            # Controlamos la velocidad de emision para simular throughput real
            time.sleep(random.uniform(0.01, 0.04))
