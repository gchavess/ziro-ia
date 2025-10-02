from flask import Flask, request, jsonify
import json
import os
import re
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras
import faiss
import numpy as np
from google import genai

# =====================
# Configurações
# =====================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY não encontrada no .env")

client = genai.Client(api_key=API_KEY)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "ziro_db")
DB_USER = os.getenv("DB_USER", "ziro_user")
DB_PASS = os.getenv("DB_PASS", "ziro_pass")
DB_PORT = os.getenv("DB_PORT", "5432")

# Limite de tokens no prompt (default 2000 se não definido)
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", 2000))

app = Flask(__name__)

# =====================
# Utilitário para limitar tokens
# =====================
def limitar_tokens(texto, max_tokens=MAX_PROMPT_TOKENS):
    # Aproximação: 1 token ≈ 4 caracteres
    max_chars = max_tokens * 4
    return texto[:max_chars]

# =====================
# Buscar histórico financeiro
# =====================
def buscar_historico():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT id, fato, causa, acao FROM public.analise_financeira;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"Erro ao buscar histórico: {e}")
        return []

# =====================
# Criar embeddings + índice FAISS
# =====================
def criar_indice(dados):
    documentos = []
    for d in dados:
        fato = d.get('fato') or ""
        causa = d.get('causa') or ""
        acao = d.get('acao') or ""
        doc = f"FATO: {fato}\nCAUSA: {causa}\nAÇÃO: {acao}".strip()
        if doc.replace("\n", "").strip():
            documentos.append(doc)

    if not documentos:
        return None, None, None

    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=documentos
    )
    embeddings_list = [e.values for e in resp.embeddings]
    embeddings = np.array(embeddings_list, dtype='float32')

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, documentos, embeddings

# =====================
# Gerar descrição a partir de gráfico
# =====================
def gerar_descricao_do_grafico(novo_dado, top_n=3, tratar_zero_como_nulo=False):
    months = novo_dado.get("months", []) or []
    datasets = novo_dado.get("datasets", []) or []
    if not datasets:
        return ""

    summaries = []
    for ds in datasets:
        label = ds.get("label", "serie")
        data = ds.get("data", []) or []
        if not data:
            continue

        arr = np.array(data, dtype=float)
        if tratar_zero_como_nulo:
            arr = np.array([np.nan if v == 0 else v for v in arr], dtype=float)

        n = len(arr)
        s = float(np.nansum(arr))
        mean = float(np.nanmean(arr))
        mn = float(np.nanmin(arr))
        mx = float(np.nanmax(arr))
        idx_min = [i for i, v in enumerate(arr) if v == mn]
        idx_max = [i for i, v in enumerate(arr) if v == mx]
        first = float(arr[0]) if n > 0 else 0.0
        last = float(arr[-1]) if n > 0 else 0.0

        try:
            slope = float(np.polyfit(np.arange(n), np.nan_to_num(arr, 0.0), 1)[0])
        except Exception:
            slope = 0.0

        pct_change = (last - first) / abs(first) if first != 0 else None

        odd_mean = float(np.nanmean(arr[0::2])) if n >= 2 else float(np.nanmean(arr))
        even_mean = float(np.nanmean(arr[1::2])) if n >= 2 else float(np.nanmean(arr))
        seasonality = None
        if not np.isnan(odd_mean) and not np.isnan(even_mean):
            if abs(odd_mean - even_mean) / (abs((odd_mean + even_mean) / 2) + 1e-9) > 0.15:
                seasonality = "sugere sazonalidade (ímpares > pares)" if odd_mean > even_mean else "sugere sazonalidade (pares > ímpares)"

        def fmt(v):
            return f"R$ {v:,.0f}".replace(',', '.')

        max_months = ", ".join([months[i] if i < len(months) else f"m{i+1}" for i in idx_max]) if idx_max else ""
        min_months = ", ".join([months[i] if i < len(months) else f"m{i+1}" for i in idx_min]) if idx_min else ""

        summaries.append({
            "label": label,
            "sum": s, "sum_str": fmt(s),
            "mean": mean, "mean_str": fmt(mean),
            "min": mn, "min_months": min_months,
            "max": mx, "max_months": max_months,
            "first": first, "last": last, "last_str": fmt(last),
            "slope": slope, "pct_change": pct_change,
            "seasonality": seasonality
        })

    if not summaries:
        return ""

    summaries_sorted = sorted(summaries, key=lambda x: abs(x["sum"]), reverse=True)
    chosen = summaries_sorted[:max(1, min(top_n, len(summaries_sorted)))]

    partes = []
    for s in chosen:
        trend = "crescente" if s['slope'] > 0 else ("decrescente" if s['slope'] < 0 else "estável")
        parte = (
            f"{s['label']}: média {s['mean_str']}, total {s['sum_str']}, tendência {trend}, "
            f"último {s['last_str']}"
        )
        if s['max_months']:
            parte += f", pico em {s['max_months']}"
        if s['min_months']:
            parte += f", vale mínimo em {s['min_months']}"
        if s['seasonality']:
            parte += f", {s['seasonality']}"
        partes.append(parte + ".")
    descricao = " | ".join(partes)
    return descricao

# =====================
# Buscar registros relevantes
# =====================
def buscar_relevantes(texto, index, documentos, embeddings, top_k=3):
    if not index or not texto.strip():
        return []

    query_resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[texto]
    )
    query_emb = np.array([query_resp.embeddings[0].values], dtype='float32')
    D, I = index.search(query_emb, top_k)

    relevantes = [documentos[i] for i in I[0] if i < len(documentos) and i >= 0]
    return relevantes

# =====================
# Gerar insights
# =====================
def gerar_insights(novo_dado, index=None, documentos=None, embeddings=None):
    descricao = (novo_dado.get('descricao') or "").strip()
    if not descricao:
        descricao = gerar_descricao_do_grafico(novo_dado, top_n=3)

    contexto = ""
    inferencia = True

    if index and documentos and embeddings is not None:
        docs_relevantes = buscar_relevantes(descricao, index, documentos, embeddings)

        if docs_relevantes:
            # Limita cada documento a 500 tokens
            docs_truncados = [limitar_tokens(doc, max_tokens=500) for doc in docs_relevantes]
            contexto = "\n\n".join(docs_truncados)
            # Limita contexto total ao definido no .env
            contexto = limitar_tokens(contexto, max_tokens=MAX_PROMPT_TOKENS)
            inferencia = False
        else:
            contexto = "Nenhum histórico relevante encontrado."

    prompt = f"""
Você é um analista financeiro experiente.

Contexto histórico:
{contexto}

Novo dado a analisar:
{json.dumps(novo_dado, ensure_ascii=False)}

Instruções:
1️⃣ Gere múltiplos insights financeiros distintos (3 a 5) com:
   - fato
   - causa
   - ação
2️⃣ Use apenas informações do contexto histórico ou do novo dado.
3️⃣ Se não houver histórico suficiente, use raciocínio próprio e marque 'inferencia': true.
4️⃣ Responda APENAS com JSON no formato:

[
  {{
    "fato": "...",
    "causa": ["...", "..."],
    "acao": ["...", "..."],
    "inferencia": true|false
  }}
]
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    result_text = response.text.strip()
    json_match = re.search(r"```json\n(.*?)```", result_text, re.DOTALL)
    json_str_cleaned = json_match.group(1) if json_match else result_text

    try:
        analise_json = json.loads(json_str_cleaned)
        if not isinstance(analise_json, list):
            raise TypeError("Resposta do Gemini não é uma lista JSON.")
        if inferencia:
            for item in analise_json:
                item['inferencia'] = True
        return analise_json
    except Exception:
        return {"erro": "Falha ao processar JSON do Gemini", "resposta_bruta": result_text}

# =====================
# Endpoint Flask
# =====================
@app.route('/analise-financeira', methods=['POST'])
def api_analise_financeira():
    if not request.json:
        return jsonify({"erro": "O corpo da requisição deve ser JSON."}), 400

    novo_dado = request.json
    historico = buscar_historico()
    index, documentos, embeddings = criar_indice(historico)
    insights = gerar_insights(novo_dado, index, documentos, embeddings)
    return jsonify(insights), 200

# =====================
# Executa Flask
# =====================
if __name__ == '__main__':
    app.run(debug=True)
