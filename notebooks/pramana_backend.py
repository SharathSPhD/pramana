"""
Pramana Backend Abstraction Module

Provides a unified interface for running inference with Pramana models across multiple backends:
- Transformers (local GPU/CPU via HuggingFace)
- HuggingFace Inference API (serverless)
- Ollama (local inference server)
- llama.cpp (direct GGUF loading)
- OpenWebUI (OpenAI-compatible API)

Usage:
    from pramana_backend import create_backend, STAGE_CONFIGS, build_user_prompt
    
    backend = create_backend("transformers", model_id="qbz506/nyaya-llama-3b-stage0-full")
    stage_config = STAGE_CONFIGS["Stage 0"]
    problem = "If P then Q. P is true. What is Q?"
    user_prompt = build_user_prompt(problem)
    output = backend.generate(user_prompt, stage_config.system_prompt)
"""

import os
import sys
import json
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any, Protocol
import math
from functools import lru_cache

# ============================================================================
# STAGE CONFIGURATIONS
# ============================================================================

FORMAT_INSTRUCTIONS = """You MUST follow the exact markdown structure below. Do NOT add "Phase" labels or alternative headings. Use the exact headers and field labels shown.

Required section order:
1) ## Samshaya (Doubt Analysis)
2) ## Pramana (Sources of Knowledge)
3) ## Pancha Avayava (5-Member Syllogism)
4) ## Tarka (Counterfactual Reasoning)
5) ## Hetvabhasa (Fallacy Check)
6) ## Nirnaya (Ascertainment)

CRITICAL:
- Your response MUST start with: "## Samshaya (Doubt Analysis)"
- Copy the template exactly and fill in every field.
- Do not add any text before the first header or after the final field.
"""

FORMAT_TEMPLATE = """## Samshaya (Doubt Analysis)
**Doubt Type**:
**Justification**:

---

## Pramana (Sources of Knowledge)
### Pratyaksha (Direct Perception)
- 
### Anumana (Inference)
- 
### Upamana (Comparison)
- 
### Shabda (Testimony)
- 

---

## Pancha Avayava (5-Member Syllogism)
### Syllogism 1: 
**Pratijna (Thesis)**:
**Hetu (Reason)**:
**Udaharana (Universal + Example)**:
**Upanaya (Application)**:
**Nigamana (Conclusion)**:

---

## Tarka (Counterfactual Reasoning)
**Hypothesis**:
**Consequence**:
**Analysis**:
**Resolution**:

---

## Hetvabhasa (Fallacy Check)
Check for Savyabhichara: 
Check for Viruddha: 
Check for Asiddha: 
Check for Satpratipaksha: 
Check for Badhita: 

---

## Nirnaya (Ascertainment)
**Final Answer**:
**Justification**:
**Confidence**:
"""

@dataclass
class StageConfig:
    """Configuration for a training stage with base and tuned model IDs."""
    name: str
    base_model_id: str
    tuned_model_id: str
    dataset_repo_id: str
    system_prompt: str
    default_max_tokens: int
    description: str

SYSTEM_PROMPT_DEFAULT = (
    "You are a Nyaya reasoning engine. Follow the exact output format provided.\n\n"
    "Use the exact section headers:\n"
    "## Samshaya (Doubt Analysis)\n"
    "## Pramana (Sources of Knowledge)\n"
    "## Pancha Avayava (5-Member Syllogism)\n"
    "## Tarka (Counterfactual Reasoning)\n"
    "## Hetvabhasa (Fallacy Check)\n"
    "## Nirnaya (Ascertainment)"
)

STAGE_CONFIGS = {
    "Stage 0": StageConfig(
        name="Stage 0 (Llama 3B)",
        base_model_id="unsloth/llama-3.2-3b-instruct",
        tuned_model_id="qbz506/nyaya-llama-3b-stage0-full",
        dataset_repo_id="qbz506/pramana-nyaya-stage0",
        system_prompt=SYSTEM_PROMPT_DEFAULT,
        default_max_tokens=2048,
        description="Fast, lightweight baseline using a 3B model."
    ),
    "Stage 1": StageConfig(
        name="Stage 1 (DeepSeek 8B)",
        base_model_id="unsloth/DeepSeek-R1-Distill-Llama-8B",
        tuned_model_id="qbz506/nyaya-deepseek-8b-stage1-full",
        dataset_repo_id="qbz506/pramana-nyaya-stage1",
        system_prompt=SYSTEM_PROMPT_DEFAULT,
        default_max_tokens=2048,
        description="Higher-capacity 8B model for improved reasoning quality."
    ),
}

def build_user_prompt(problem: str) -> str:
    """Build the user-visible prompt with format requirements."""
    return f"""### Problem:
{problem}

### Instructions:
{FORMAT_INSTRUCTIONS}

### Template:
{FORMAT_TEMPLATE}

### Nyaya Reasoning:
"""

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_package(package: str, quiet: bool = True) -> None:
    """Install a Python package using pip."""
    cmd = [sys.executable, "-m", "pip", "install", package]
    if quiet:
        cmd.append("-q")
    subprocess.check_call(cmd)

# ============================================================================
# GENERATION PARAMETER NORMALIZATION
# ============================================================================

MAX_NEW_TOKENS_CAP = 4096
TOP_K_CAP = 200
MAX_TEMPERATURE = 1.5

def normalize_generation_params(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Dict[str, Any]:
    """Normalize generation parameters to valid ranges."""
    max_new_tokens = int(max(1, min(max_new_tokens, MAX_NEW_TOKENS_CAP)))
    temperature = float(max(0.0, min(temperature, MAX_TEMPERATURE)))
    top_p = float(max(0.0, min(top_p, 1.0)))
    top_k = int(max(0, min(top_k, TOP_K_CAP)))
    do_sample = temperature > 0.0
    
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
    }

# ============================================================================
# BACKEND PROTOCOL
# ============================================================================

class Backend(Protocol):
    """Protocol for all backend implementations."""
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate text from the model."""
        ...

# ============================================================================
# BACKEND IMPLEMENTATIONS
# ============================================================================

class TransformersBackend:
    """Backend using HuggingFace transformers library (local GPU/CPU)."""
    
    def __init__(self, model_id: str, cache_model: bool = True, hf_token: Optional[str] = None):
        """Initialize Transformers backend.
        
        Args:
            model_id: HuggingFace model identifier
            cache_model: Whether to cache loaded models (default: True)
            hf_token: Optional HuggingFace token for gated models
        """
        self.model_id = model_id
        self.cache_model = cache_model
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._model = None
        self._tokenizer = None
        self._device = None
        
    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return
            
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast
        except ImportError as e:
            raise RuntimeError(
                "transformers and torch are required. Install with: pip install transformers torch"
            ) from e
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, token=self.hf_token, use_fast=True
            )
        except ValueError as exc:
            if "TokenizersBackend" in str(exc):
                tokenizer = LlamaTokenizerFast.from_pretrained(
                    self.model_id, token=self.hf_token
                )
            else:
                raise
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=self.hf_token,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        
        if self.cache_model:
            self._model = model
            self._tokenizer = tokenizer
            self._device = device
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate text using transformers."""
        import torch
        
        self._load_model()
        
        if self.cache_model:
            model = self._model
            tokenizer = self._tokenizer
            device = self._device
        else:
            # Reload for non-cached mode
            self._load_model()
            model = self._model
            tokenizer = self._tokenizer
            device = self._device
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        params = normalize_generation_params(max_new_tokens, temperature, top_p, top_k)
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=params["max_new_tokens"],
                do_sample=params["do_sample"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"],
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        
        if not self.cache_model:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        return text


class HFInferenceBackend:
    """Backend using HuggingFace Inference API (serverless)."""
    
    def __init__(self, model_id: str, hf_token: Optional[str] = None):
        """Initialize HF Inference backend.
        
        Args:
            model_id: HuggingFace model identifier
            hf_token: Optional HuggingFace token (required for gated models)
        """
        self.model_id = model_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            ) from e
        
        self.client = InferenceClient(model=model_id, token=self.hf_token)
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate text using HF Inference API."""
        params = normalize_generation_params(max_new_tokens, temperature, top_p, top_k)
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = self.client.text_generation(
            full_prompt,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"] if params["top_k"] > 0 else None,
            return_full_text=False,
        )
        
        return response


class OllamaBackend:
    """Backend using Ollama REST API (localhost:11434)."""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        """Initialize Ollama backend.
        
        Args:
            model_name: Name of the Ollama model (e.g., "llama3.2:3b")
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        
        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "requests is required. Install with: pip install requests"
            ) from e
        
        self.requests = requests
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate text using Ollama API."""
        params = normalize_generation_params(max_new_tokens, temperature, top_p, top_k)
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": params["max_new_tokens"],
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "top_k": params["top_k"] if params["top_k"] > 0 else None,
            },
        }
        
        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except self.requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}") from e


class LlamaCppBackend:
    """Backend using llama-cpp-python (direct GGUF loading) or HTTP server mode."""
    
    def __init__(self, model_path: Optional[str] = None, server_url: Optional[str] = None):
        """Initialize LlamaCpp backend.
        
        Args:
            model_path: Path to GGUF model file (for direct loading)
            server_url: URL to llama.cpp HTTP server (e.g., "http://localhost:8080")
                       If provided, uses HTTP mode instead of direct loading.
        """
        if not model_path and not server_url:
            raise ValueError("Either model_path or server_url must be provided")
        
        self.model_path = model_path
        self.server_url = server_url.rstrip("/") if server_url else None
        
        if self.server_url:
            # HTTP server mode
            try:
                import requests
            except ImportError as e:
                raise RuntimeError(
                    "requests is required for HTTP mode. Install with: pip install requests"
                ) from e
            self.requests = requests
            self._llama = None
        else:
            # Direct loading mode
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise RuntimeError(
                    "llama-cpp-python is required. Install with: pip install llama-cpp-python"
                ) from e
            self._llama = Llama(model_path=model_path, n_ctx=4096, verbose=False)
            self.requests = None
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate text using llama.cpp."""
        params = normalize_generation_params(max_new_tokens, temperature, top_p, top_k)
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        if self.server_url:
            # HTTP server mode
            payload = {
                "prompt": full_prompt,
                "n_predict": params["max_new_tokens"],
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "top_k": params["top_k"] if params["top_k"] > 0 else -1,
                "stream": False,
            }
            
            try:
                response = self.requests.post(
                    f"{self.server_url}/completion",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
                return result.get("content", "")
            except self.requests.exceptions.RequestException as e:
                raise RuntimeError(f"llama.cpp HTTP server request failed: {e}") from e
        else:
            # Direct loading mode
            response = self._llama(
                full_prompt,
                max_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=params["top_k"] if params["top_k"] > 0 else -1,
            )
            return response["choices"][0]["text"]


class OpenWebUIBackend:
    """Backend using OpenAI-compatible API (e.g., OpenWebUI, Ollama's OpenAI endpoint)."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize OpenWebUI backend.
        
        Args:
            base_url: Base URL for OpenAI-compatible API (default: http://localhost:11434)
            api_key: Optional API key for authentication
            model_name: Model name to use in API calls (e.g., "llama3.2:3b")
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name or ""
        
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai is required. Install with: pip install openai"
            ) from e
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or "ollama",  # Default for local Ollama
        )
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate text using OpenAI-compatible API."""
        params = normalize_generation_params(max_new_tokens, temperature, top_p, top_k)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"] if params["top_p"] < 1.0 else None,
            )
            
            if not response.choices:
                raise RuntimeError("OpenAI-compatible API returned no choices")
            
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible API request failed: {e}") from e


# ============================================================================
# SELF-CONTAINED VALIDATION & SCORING (no external dependencies)
# ============================================================================

def parse_nyaya_phases(text: str) -> Dict[str, Optional[str]]:
    """Parse 6 Nyaya phases from model output using regex.
    
    Returns dict with keys: samshaya, pramana, pancha_avayava, tarka, hetvabhasa, nirnaya.
    Values are the section text (str) or None if not found.
    """
    import re as _re
    phases = {}
    patterns = {
        "samshaya": r"##\s+Samshaya.*?\n(.*?)(?=\n##\s+|\Z)",
        "pramana": r"##\s+Pramana.*?\n(.*?)(?=\n##\s+|\Z)",
        "pancha_avayava": r"##\s+Pancha Avayava.*?\n(.*?)(?=\n##\s+|\Z)",
        "tarka": r"##\s+Tarka.*?\n(.*?)(?=\n##\s+|\Z)",
        "hetvabhasa": r"##\s+Hetvabhasa.*?\n(.*?)(?=\n##\s+|\Z)",
        "nirnaya": r"##\s+Nirnaya.*?\n(.*?)(?=\n##\s+|\Z)",
    }
    for phase_name, pattern in patterns.items():
        match = _re.search(pattern, text, _re.DOTALL | _re.IGNORECASE)
        phases[phase_name] = match.group(1).strip() if match else None
    return phases


def validate_structure(text: str) -> Dict[str, Any]:
    """Tier 1 structural validation of Nyaya output (self-contained, regex-based).
    
    Returns:
        dict with keys: passed (bool), phases_present (int), errors (list[str]),
        phase_details (dict of phase -> bool), pramana_sources (int), syllogism_count (int)
    """
    import re as _re
    errors = []
    phase_details = {}

    required_phases = [
        "Samshaya", "Pramana", "Pancha Avayava", "Tarka", "Hetvabhasa", "Nirnaya"
    ]
    for phase in required_phases:
        found = bool(_re.search(rf"##\s+{_re.escape(phase)}", text, _re.IGNORECASE))
        phase_details[phase] = found
        if not found:
            errors.append(f"Missing phase: {phase}")

    phases_present = sum(phase_details.values())

    # Check Pramana sources
    pramana_sources = 0
    pramana_match = _re.search(
        r"##\s+Pramana.*?\n(.*?)(?=\n##\s+|\Z)", text, _re.DOTALL | _re.IGNORECASE
    )
    if pramana_match:
        pramana_text = pramana_match.group(1)
        for src in ["Pratyaksha", "Anumana", "Upamana", "Shabda"]:
            if _re.search(rf"###\s+{_re.escape(src)}", pramana_text, _re.IGNORECASE):
                pramana_sources += 1
        if pramana_sources == 0:
            errors.append("Pramana section has no knowledge sources (Pratyaksha/Anumana/Upamana/Shabda)")

    # Check Pancha Avayava syllogisms
    syllogism_count = 0
    pancha_match = _re.search(
        r"##\s+Pancha Avayava.*?\n(.*?)(?=\n##\s+|\Z)", text, _re.DOTALL | _re.IGNORECASE
    )
    if pancha_match:
        pancha_text = pancha_match.group(1)
        syllogism_count = len(_re.findall(r"###\s+Syllogism", pancha_text, _re.IGNORECASE))
        if syllogism_count == 0:
            errors.append("Pancha Avayava has no syllogisms")
        else:
            syllogisms = _re.finditer(
                r"###\s+Syllogism.*?\n(.*?)(?=###\s+Syllogism|##\s+|\Z)",
                pancha_text, _re.DOTALL | _re.IGNORECASE,
            )
            for idx, sm in enumerate(syllogisms, 1):
                st = sm.group(1)
                for member in ["Pratijna", "Hetu", "Udaharana", "Upanaya", "Nigamana"]:
                    if not _re.search(rf"\*\*{_re.escape(member)}.*?\*\*:", st, _re.IGNORECASE):
                        errors.append(f"Syllogism {idx} missing member: {member}")

    return {
        "passed": len(errors) == 0,
        "phases_present": phases_present,
        "errors": errors,
        "phase_details": phase_details,
        "pramana_sources": pramana_sources,
        "syllogism_count": syllogism_count,
    }


def score_content_quality(text: str, problem: str) -> Dict[str, float]:
    """Tier 2 content quality scoring (self-contained, regex-based).
    
    Scores: pratyaksha_grounding, udaharana_valid, tarka_meaningful,
    hetvabhasa_completeness, overall.
    """
    import re as _re
    scores: Dict[str, float] = {}
    phases = parse_nyaya_phases(text)

    # 1. Pratyaksha grounding: do claims reference problem terms?
    pratyaksha_score = 0.0
    if phases.get("pramana"):
        pm = _re.search(
            r"###\s+Pratyaksha.*?\n(.*?)(?=###|##|\Z)",
            phases["pramana"], _re.DOTALL | _re.IGNORECASE,
        )
        if pm:
            claims = [l.strip("- ").strip() for l in pm.group(1).split("\n") if l.strip().startswith("-")]
            if claims:
                problem_lower = problem.lower()
                grounded = sum(
                    1 for c in claims
                    if any(w in problem_lower for w in c.lower().split()[:5] if len(w) > 2)
                )
                pratyaksha_score = grounded / len(claims)
    scores["pratyaksha_grounding"] = pratyaksha_score

    # 2. Udaharana: universal rule phrasing
    udaharana_valid = False
    if phases.get("pancha_avayava"):
        for pat in [r"\bwherever\b.+,\s*.+", r"\bwhenever\b.+", r"\bin all cases\b.+",
                    r"\bfor any\b.+\bif\b.+\bthen\b.+", r"\bfor example\b"]:
            if _re.search(pat, phases["pancha_avayava"], _re.IGNORECASE):
                udaharana_valid = True
                break
    scores["udaharana_valid"] = 1.0 if udaharana_valid else 0.0

    # 3. Tarka: counterfactual with negation + contradiction
    tarka_meaningful = False
    if phases.get("tarka"):
        tl = phases["tarka"].lower()
        hyp_match = _re.search(r"\*\*Hypothesis\*\*:\s*(.+?)(?=\*\*|\Z)", phases["tarka"], _re.DOTALL | _re.IGNORECASE)
        has_neg = False
        if hyp_match:
            has_neg = any(w in hyp_match.group(1).lower() for w in ["not", "no", "never", "suppose", "assume", "contrary"])
        has_contra = any(w in tl for w in ["contradiction", "contradicts", "impossible", "cannot", "absurd", "violat"])
        tarka_meaningful = has_neg and has_contra
    scores["tarka_meaningful"] = 1.0 if tarka_meaningful else 0.0

    # 4. Hetvabhasa: fallacy type coverage
    hetvabhasa_completeness = 0.0
    if phases.get("hetvabhasa"):
        fallacy_types = ["savyabhichara", "viruddha", "prakaranasama", "sadhyasama",
                         "asiddha", "satpratipaksha", "badhita", "kalaatita"]
        hl = phases["hetvabhasa"].lower()
        found = sum(1 for ft in fallacy_types if ft in hl)
        hetvabhasa_completeness = found / max(len(fallacy_types), 1)
    scores["hetvabhasa_completeness"] = hetvabhasa_completeness

    scores["overall"] = sum(scores.values()) / len(scores)
    return scores


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the Final Answer from the Nirnaya section."""
    import re as _re
    phases = parse_nyaya_phases(text)
    nirnaya = phases.get("nirnaya", "")
    if nirnaya:
        m = _re.search(r"\*\*Final Answer\*\*:\s*(.+?)(?=\*\*|\n\n|\Z)", nirnaya, _re.DOTALL | _re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


# ---- Answer Scoring (adapted from scoring.py, no external deps) ----

def normalize_text(text: str) -> str:
    """Normalize text for comparisons (lowercase, remove punctuation)."""
    import re as _re
    cleaned = _re.sub(r"[^a-z0-9\s]+", " ", text.lower())
    return _re.sub(r"\s+", " ", cleaned).strip()


def token_overlap_ratio(predicted: str, ground_truth: str) -> float:
    """Token overlap ratio relative to ground truth tokens."""
    pred_tokens = set(normalize_text(predicted).split())
    gt_tokens = set(normalize_text(ground_truth).split())
    if not gt_tokens:
        return 0.0
    return len(pred_tokens & gt_tokens) / len(gt_tokens)


def score_answers(
    predicted: str,
    ground_truth: str,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """Score predicted vs ground truth answer (no external ML deps).
    
    Returns dict with: exact_match, normalized_match, token_overlap, semantic_match.
    Uses token overlap as lightweight semantic similarity proxy.
    """
    exact_match = predicted.strip().lower() == ground_truth.strip().lower()
    norm_pred = normalize_text(predicted)
    norm_gt = normalize_text(ground_truth)
    normalized_match = norm_gt in norm_pred if norm_gt else False
    overlap = token_overlap_ratio(predicted, ground_truth)
    return {
        "exact_match": exact_match,
        "normalized_match": normalized_match,
        "token_overlap": overlap,
        "semantic_match": overlap >= threshold,
        "semantic_similarity": overlap,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for a proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p_hat = successes / total
    denom = 1 + (z * z) / total
    center = (p_hat + (z * z) / (2 * total)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) / total) + (z * z) / (4 * total * total)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


# ============================================================================
# EMBEDDED EXAMPLE PROBLEMS (no external files needed)
# ============================================================================

EXAMPLE_PROBLEMS = [
    {
        "id": "pramana-001",
        "problem_type": "constraint_satisfaction",
        "difficulty": "medium",
        "problem": (
            "Three people (Alice, Bob, Carol) each have one pet: a cat, a dog, or a fish.\n\n"
            "**Constraints**:\n"
            "1. Alice does not have the cat\n"
            "2. Bob has the dog\n"
            "3. Carol does not have the fish\n\n"
            "**Question**: Who has which pet?"
        ),
        "ground_truth": "Alice has the fish, Bob has the dog, Carol has the cat",
    },
    {
        "id": "pramana-003",
        "problem_type": "transitive_reasoning",
        "difficulty": "medium",
        "problem": (
            "Four friends (Alice, Bob, Carol, David) are comparing their heights.\n\n"
            "**Constraints**:\n"
            "1. Alice is taller than Bob\n"
            "2. Bob is taller than Carol\n"
            "3. Carol is taller than David\n"
            "4. Alice is taller than Carol\n\n"
            "**Question**: What is the complete height ranking from tallest to shortest?"
        ),
        "ground_truth": "Ranking: Alice > Bob > Carol > David (Alice is tallest, David is shortest)",
    },
    {
        "id": "pramana-005",
        "problem_type": "multi_step_deduction",
        "difficulty": "medium",
        "problem": (
            "Consider four logical statements P, Q, R, and S.\n\n"
            "**Given Facts**:\n"
            "1. If P is true, then Q is true\n"
            "2. If Q is true, then R is true\n"
            "3. If R is true, then S is true\n"
            "4. P is true\n\n"
            "**Question**: What are the truth values of P, Q, R, and S?"
        ),
        "ground_truth": "All four statements are true: P is true, Q is true, R is true, S is true",
    },
    {
        "id": "test-001",
        "problem_type": "constraint_satisfaction",
        "difficulty": "easy",
        "problem": (
            "Three boxes (A, B, C) contain one item each: a key, a coin, or a ring.\n\n"
            "**Constraints**:\n"
            "1. The key is not in box A\n"
            "2. The coin is in box B\n"
            "3. The ring is not in box B\n\n"
            "**Question**: Which item is in which box?"
        ),
        "ground_truth": "Box A has the ring, Box B has the coin, Box C has the key",
    },
    {
        "id": "test-004",
        "problem_type": "boolean_sat",
        "difficulty": "medium",
        "problem": (
            "Given boolean variables X, Y, Z:\n\n"
            "**Constraints**:\n"
            "1. X OR Y is true\n"
            "2. NOT X OR Z is true\n"
            "3. Y AND Z is false\n"
            "4. X is true\n\n"
            "**Question**: What are the values of X, Y, and Z?"
        ),
        "ground_truth": "X is true, Y is false, Z is true",
    },
]


# ============================================================================
# COLAB / STANDALONE SETUP HELPERS
# ============================================================================

def setup_ollama(model_name: str = "nyaya-llama-3b-stage0",
                 gguf_url: str = "https://huggingface.co/qbz506/nyaya-llama-3b-stage0-full/resolve/main/nyaya-llama-3b-stage0-merged-q4.gguf") -> str:
    """Install Ollama, download GGUF, create model. Works on Colab/Linux.
    
    Returns the model name to use with OllamaBackend.
    """
    import shutil
    
    # 1. Install Ollama if not present
    if not shutil.which("ollama"):
        print("Installing Ollama...")
        subprocess.run("apt-get update -qq && apt-get install -y -qq zstd curl", 
                       shell=True, capture_output=True)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                       shell=True, capture_output=True)
        print("  Ollama installed")
    
    # 2. Start server if not running
    import time
    try:
        import requests as _req
        _req.get("http://localhost:11434/api/tags", timeout=2)
    except Exception:
        print("Starting Ollama server...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={**os.environ, "OLLAMA_HOST": "0.0.0.0"},
        )
        time.sleep(3)
    
    # 3. Check if model already exists
    try:
        import requests as _req
        resp = _req.get("http://localhost:11434/api/tags", timeout=5)
        existing = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        if model_name in existing:
            print(f"  Model '{model_name}' already available")
            return model_name
    except Exception:
        pass
    
    # 4. Download GGUF
    gguf_path = f"/tmp/{model_name}.gguf"
    if not os.path.exists(gguf_path):
        print(f"Downloading GGUF model (~1.9 GB)...")
        subprocess.run(
            ["curl", "-fSL", "-o", gguf_path, gguf_url],
            check=True,
        )
        print("  Download complete")
    
    # 5. Create Modelfile and import
    modelfile = f"""FROM {gguf_path}
SYSTEM \"\"\"
You are a Nyaya reasoning engine. Follow the exact output format provided.
Use the exact section headers:
## Samshaya (Doubt Analysis)
## Pramana (Sources of Knowledge)
## Pancha Avayava (5-Member Syllogism)
## Tarka (Counterfactual Reasoning)
## Hetvabhasa (Fallacy Check)
## Nirnaya (Ascertainment)
\"\"\"
PARAMETER temperature 0
PARAMETER num_ctx 4096
"""
    modelfile_path = "/tmp/Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile)
    
    print(f"Creating Ollama model '{model_name}'...")
    subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], check=True)
    print(f"  Model '{model_name}' ready")
    return model_name


def download_gguf(
    model_name: str = "nyaya-llama-3b-stage0",
    url: str = "https://huggingface.co/qbz506/nyaya-llama-3b-stage0-full/resolve/main/nyaya-llama-3b-stage0-merged-q4.gguf",
    dest_dir: str = "/tmp",
) -> str:
    """Download a GGUF model for llama.cpp. Returns local path."""
    dest_path = os.path.join(dest_dir, f"{model_name}.gguf")
    if os.path.exists(dest_path):
        print(f"GGUF already at {dest_path}")
        return dest_path
    print(f"Downloading GGUF to {dest_path} (~1.9 GB)...")
    subprocess.run(["curl", "-fSL", "-o", dest_path, url], check=True)
    print("  Download complete")
    return dest_path


def load_test_problems(source: str = "embedded") -> list:
    """Load test problems from various sources.
    
    Args:
        source: "embedded" (built-in), "huggingface" (download from HF datasets),
                or a directory path containing .md files.
    
    Returns:
        List of dicts with keys: id, problem_type, difficulty, problem, ground_truth
    """
    if source == "embedded":
        return list(EXAMPLE_PROBLEMS)
    
    if source == "huggingface":
        try:
            from datasets import load_dataset
            ds = load_dataset(STAGE_CONFIGS["Stage 0"].dataset_repo_id, split="test",
                              token=os.environ.get("HF_TOKEN"))
            problems = []
            for row in ds:
                problems.append({
                    "id": row.get("id", f"hf-{len(problems)}"),
                    "problem_type": row.get("problem_type", "unknown"),
                    "difficulty": row.get("difficulty", "unknown"),
                    "problem": row.get("instruction", row.get("problem", "")),
                    "ground_truth": row.get("ground_truth", ""),
                })
            return problems
        except Exception as e:
            print(f"Could not load from HuggingFace: {e}")
            print("Falling back to embedded examples")
            return list(EXAMPLE_PROBLEMS)
    
    # Assume source is a directory path
    from pathlib import Path
    problems = []
    src_dir = Path(source)
    if src_dir.is_dir():
        for md_file in sorted(src_dir.glob("*.md")):
            if md_file.name == ".gitkeep":
                continue
            content = md_file.read_text(encoding="utf-8")
            import re as _re
            # Lightweight frontmatter extraction
            fm_match = _re.match(r"^---\s*\n(.*?)^---\s*\n(.*)$", content, _re.DOTALL | _re.MULTILINE)
            if fm_match:
                import yaml
                try:
                    meta = yaml.safe_load(fm_match.group(1))
                except Exception:
                    meta = {}
                body = fm_match.group(2)
                # Extract problem section
                prob_match = _re.search(r"^#\s+Problem\s*\n(.*?)(?=^##\s+|\Z)", body, _re.MULTILINE | _re.DOTALL)
                problem_text = prob_match.group(1).strip() if prob_match else ""
                problems.append({
                    "id": meta.get("id", md_file.stem),
                    "problem_type": meta.get("problem_type", "unknown"),
                    "difficulty": meta.get("difficulty", "unknown"),
                    "problem": problem_text,
                    "ground_truth": meta.get("ground_truth", ""),
                })
    return problems if problems else list(EXAMPLE_PROBLEMS)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_backend(
    backend_type: str,
    model_id: Optional[str] = None,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    server_url: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    cache_model: bool = True,
    hf_token: Optional[str] = None,
    **kwargs,
) -> Backend:
    """Create a backend instance based on type.
    
    Args:
        backend_type: One of "transformers", "hf_inference", "ollama", "llamacpp", "openwebui"
        model_id: HuggingFace model ID (for transformers/hf_inference)
        model_path: Path to GGUF model file (for llamacpp direct mode)
        model_name: Ollama model name (for ollama backend)
        server_url: URL to llama.cpp HTTP server (for llamacpp HTTP mode)
        base_url: Base URL for OpenAI-compatible API (for openwebui)
        api_key: API key for authentication (for openwebui)
        cache_model: Whether to cache models (for transformers, default: True)
        hf_token: HuggingFace token (for transformers/hf_inference)
    
    Returns:
        Backend instance
    
    Raises:
        ValueError: If backend_type is invalid or required parameters are missing
    """
    backend_type = backend_type.lower().strip()
    
    if backend_type == "transformers":
        if not model_id:
            raise ValueError("model_id is required for transformers backend")
        return TransformersBackend(
            model_id=model_id,
            cache_model=cache_model,
            hf_token=hf_token,
        )
    
    elif backend_type == "hf_inference":
        if not model_id:
            raise ValueError("model_id is required for hf_inference backend")
        return HFInferenceBackend(model_id=model_id, hf_token=hf_token)
    
    elif backend_type == "ollama":
        if not model_name:
            raise ValueError("model_name is required for ollama backend")
        return OllamaBackend(model_name=model_name, base_url=base_url or "http://localhost:11434")
    
    elif backend_type == "llamacpp":
        if server_url:
            return LlamaCppBackend(server_url=server_url)
        elif model_path:
            return LlamaCppBackend(model_path=model_path)
        else:
            raise ValueError("Either model_path or server_url is required for llamacpp backend")
    
    elif backend_type == "openwebui":
        return OpenWebUIBackend(
            base_url=base_url or "http://localhost:11434",
            api_key=api_key,
            model_name=model_name,
        )
    
    else:
        raise ValueError(
            f"Unknown backend_type: {backend_type}. "
            "Must be one of: transformers, hf_inference, ollama, llamacpp, openwebui"
        )
