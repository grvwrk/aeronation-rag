import re
import sys
from pathlib import Path

try:
    import tiktoken
except Exception:
    print('tiktoken not installed')
    raise


def parse_config(cfg_path: Path):
    cfg = {}
    text = cfg_path.read_text(encoding='utf-8')
    # simple top-level key:value parser
    for m in re.finditer(r'^(\w+):\s*(\S.*)$', text, flags=re.MULTILINE):
        key = m.group(1).strip()
        val = m.group(2).strip()
        # try to cast numbers
        if re.match(r'^\d+$', val):
            val = int(val)
        cfg[key] = val

    # extract nested model sections (OPENAI, GROQ, ANTHROPIC, OLLAMA)
    for model in ['OPENAI', 'GROQ', 'ANTHROPIC', 'OLLAMA']:
        sec = re.search(rf'{model}:\n((?:\s+.+\n)+)', text)
        if sec:
            block = sec.group(1)
            if model in cfg and not isinstance(cfg[model], dict):
                cfg[model] = {}
            for m in re.finditer(r'^\s+(\w+):\s*(.+)$', block, flags=re.MULTILINE):
                k = m.group(1).strip()
                v = m.group(2).strip()
                if re.match(r'^\d+$', v):
                    v = int(v)
                cfg.setdefault(model, {})[k] = v

    # read specific RAG keys
    for rag_key in ['RAG_CITATION_CHUNK_SIZE', 'RAG_SIMILARITY_TOP_K', 'RAG_CITATION_CHUNK_OVERLAP', 'RAG_STREAMING']:
        m = re.search(rf'^{rag_key}:\s*(.+)$', text, flags=re.MULTILINE)
        if m:
            v = m.group(1).strip()
            if v.isdigit():
                v = int(v)
            cfg[rag_key] = v

    return cfg


def load_prompt(prompts_dir: Path, name: str):
    path = prompts_dir / name
    if not path.exists():
        return ''
    return path.read_text(encoding='utf-8')


def main():
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / 'config' / 'config.yaml'
    prompts_dir = repo_root / 'prompts'

    cfg = parse_config(cfg_path)

    model_type = cfg.get('LLM_MODEL_TYPE', 'GROQ')
    model_cfg = cfg.get(model_type, {})
    model_name = model_cfg.get('MODEL_NAME', model_cfg.get('MODEL', ''))
    max_tokens = int(model_cfg.get('MAX_TOKENS', cfg.get('MAX_TOKENS', 4096)))

    chunk_size = int(cfg.get('RAG_CITATION_CHUNK_SIZE', 512))
    top_k = int(cfg.get('RAG_SIMILARITY_TOP_K', 3))
    reserved_output = max_tokens

    # load prompts
    citation = load_prompt(prompts_dir, 'citation_template.prompt')
    qa = load_prompt(prompts_dir, 'qa_template.prompt')
    refine = load_prompt(prompts_dir, 'refine_template.prompt')
    system = load_prompt(prompts_dir, 'system.prompt')

    combined_prompt = '\n'.join([citation, qa, refine, system])

    # choose encoding
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding('cl100k_base')

    prompt_tokens = len(enc.encode(combined_prompt))

    # We don't have actual chunk text here; use the configured chunk size as token estimate
    per_chunk_tokens = chunk_size

    total_retrieved_tokens = per_chunk_tokens * top_k

    window = max_tokens

    available = window - (prompt_tokens + total_retrieved_tokens + reserved_output)

    print('Model type:', model_type)
    print('Model name:', model_name)
    print('Model window (MAX_TOKENS):', window)
    print('Prompt tokens (citation+qa+refine+system):', prompt_tokens)
    print('Per-chunk token estimate (from config RAG_CITATION_CHUNK_SIZE):', per_chunk_tokens)
    print('Similarity top_k:', top_k)
    print('Total retrieved tokens (per_chunk * top_k):', total_retrieved_tokens)
    print('Reserved output tokens (model MAX_TOKENS):', reserved_output)
    print('Calculated available context size:', available)

    if available < 0:
        print('\nDEFICIT: The combined prompt+retrieved+reserved output exceeds the model window by', -available, 'tokens')
        print('Suggested quick fixes: reduce RAG_CITATION_CHUNK_SIZE, reduce RAG_SIMILARITY_TOP_K, or reduce model MAX_TOKENS reservation')


if __name__ == '__main__':
    main()
