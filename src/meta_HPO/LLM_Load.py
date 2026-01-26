#--------------------------------------
 # Load the LLM using Ollam
#---------------------------------------


import subprocess
import shutil

def _ollama_installed():
    return shutil.which("ollama") is not None


def _ollama_model_exists(model_name: str):
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        return any(model_name in line for line in out.splitlines())
    except Exception:
        return False


def _ollama_pull(model_name: str):
    print(f"⬇️ Pulling Ollama model: {model_name}")
    subprocess.run(["ollama", "pull", model_name], check=True)


def load_llm(
    backend: str = "ollama",
    model_name: str = "qwen2:3b",
    temperature: float = 0.2,
    warn_large: bool = True
):
    backend = backend.lower()

    # --------------------------
    # Ollama (BEST for CPU)
    # --------------------------
    if backend == "ollama":
        if not _ollama_installed():
            raise RuntimeError(
                "❌ Ollama not installed. Install from: https://ollama.com"
            )

        if not _ollama_model_exists(model_name):
            _ollama_pull(model_name)

        if warn_large and any(x in model_name for x in ["13b", "70b"]):
            print("⚠️ WARNING: This model may be too large for CPU.")

        from langchain_community.llms import Ollama
        return Ollama(
            model=model_name,
            temperature=temperature,
            num_ctx=4096,
            num_thread=4   # adjust based on your CPU cores
        )

    # --------------------------
    # HuggingFace (CPU fallback)
    # --------------------------
    elif backend == "hf":
        from langchain_community.llms import HuggingFaceHub

        if warn_large and any(x in model_name.lower() for x in ["13b", "70b"]):
            print("⚠️ WARNING: Large HF model on CPU will be VERY slow.")

        return HuggingFaceHub(
            repo_id=model_name,
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": 512
            }
        )


    else:
        raise ValueError("Unsupported backend: ollama | openai | hf")