from flask import Flask, request, jsonify
import json
import google.generativeai as genai
import os
import re # Importa a biblioteca para expressões regulares
from dotenv import load_dotenv

# =====================
# 1️⃣ Configurações
# =====================

# 2. Carrega as variáveis do arquivo .env
load_dotenv()

# Inicializa o Flask
app = Flask(__name__)

# 3. Lê a API Key da variável de ambiente
api_key = os.getenv("API_KEY")

# Verifica se a chave da API foi carregada corretamente
if not api_key:
    raise ValueError("A variável de ambiente 'API_KEY' não foi encontrada. Verifique seu arquivo .env.")

genai.configure(api_key=api_key)

# =====================
# 2️⃣ Endpoint da API
# =====================

@app.route('/analise-financeira', methods=['POST'])
def analise_financeira():
    """
    Endpoint que recebe dados financeiros via POST e retorna a análise do Gemini.
    """
    if not request.json:
        return jsonify({"erro": "O corpo da requisição deve ser um JSON."}), 400
    
    json_data = request.json
    
    try:
        json_str = json.dumps(json_data, indent=2)

        prompt = f"""
        Você é um analista financeiro inteligente. 
        Analise os dados financeiros a seguir:

        {json_str}

        Gere uma lista de insights financeiros relevantes. Cada insight deve ser um objeto com:
        1️⃣ Um FATO relevante que possa ser observado nos dados.
        2️⃣ Uma ou mais possíveis CAUSAS para esse fato.
        3️⃣ Uma ou mais ações sugeridas para melhorar ou corrigir a situação.

        Responda **APENAS** com uma lista de objetos no seguinte formato JSON, sem nenhum texto ou formatação adicional:

        [
          {{
            "fato": "...",
            "causa": ["...", "..."],
            "acao": ["...", "..."]
          }},
          {{
            "fato": "...",
            "causa": ["...", "..."],
            "acao": ["...", "..."]
          }}
        ]
        """

        # Chama o modelo Gemini
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)

        # Trata a resposta do Gemini
        result_text = response.text.strip()
        
        # Encontra o primeiro bloco de código JSON na resposta
        json_match = re.search(r"```json\n(.*?)```", result_text, re.DOTALL)
        if json_match:
            json_str_cleaned = json_match.group(1)
        else:
            json_str_cleaned = result_text

        analise_json = json.loads(json_str_cleaned)

        # Opcional: valida se a resposta é uma lista
        if not isinstance(analise_json, list):
            raise TypeError("A resposta do Gemini não é uma lista JSON.")

        return jsonify(analise_json), 200

    except genai.types.GenerationError as e:
        return jsonify({"erro": f"Erro do modelo Gemini: {str(e)}"}), 500
    except (json.JSONDecodeError, TypeError) as e:
        # Erro ao decodificar JSON ou tipo de dado incorreto
        return jsonify({
            "erro": f"Resposta do Gemini não está no formato JSON esperado (lista de objetos). Erro: {str(e)}",
            "resposta_bruta": result_text
        }), 500
    except Exception as e:
        return jsonify({"erro": f"Erro interno: {str(e)}"}), 500

# =====================
# 3️⃣ Executa a API
# =====================
if __name__ == '__main__':
    app.run(debug=True)
    