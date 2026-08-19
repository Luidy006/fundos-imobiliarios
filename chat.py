import logging
import os
import streamlit as st
import config

logger = logging.getLogger('FII.chat')

def get_ai_client():
    """
    Tenta retornar o cliente Gemini e, se falhar, o cliente Claude.
    """
    try:
        from google import genai
        # GEMINI_API_KEY from secrets or env
        api_key = None
        if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
            api_key = st.secrets['GEMINI_API_KEY']
        else:
            api_key = os.environ.get('GEMINI_API_KEY')
            
        if api_key:
            client = genai.Client(api_key=api_key)
            return 'gemini', client
    except ImportError:
        logger.warning("google.genai não instalado.")
    
    try:
        import anthropic
        api_key = None
        if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
            api_key = st.secrets['ANTHROPIC_API_KEY']
        else:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            
        if api_key:
            client = anthropic.Anthropic(api_key=api_key)
            return 'claude', client
    except ImportError:
        logger.warning("anthropic não instalado.")
        
    return None, None

def build_chat_context(top_df, profile, objective, horizon_months, criterio, cvm_facts=None) -> str:
    """
    Monta o contexto para a IA com base na alocação atual e fatos relevantes.
    """
    linhas = [
        f"Perfil do investidor: {profile}",
        f"Objetivo: {objective}",
        f"Horizonte: {horizon_months} meses",
        f"Critério de alocação escolhido: {criterio}",
        "",
        "Ranking atual de FIIs recomendados (do melhor para o pior colocado):",
    ]
    for t in top_df.index:
        r = top_df.loc[t]
        linhas.append(
            f"- {t} ({r.get('Segmento', '')}): Preço R$ {r.get('Preco', 0):.2f}, "
            f"P/VP {r.get('P/VP', 0):.2f}, DY {r.get('DY', 0):.1f}%, Vacância {r.get('Vacancia', 0):.1f}%, "
            f"Adequação ao perfil {r.get('Adequação', 0):.1%}, "
            f"Chance de sucesso {r.get('Chance_Sucesso', 0):.1%}, "
            f"Retorno esperado {r.get('Retorno_Esperado', 0):+.1%}, "
            f"Valor a investir R$ {r.get('Valor_Alocado', 0):.2f} ({r.get('Cotas_Estimadas', 0)} cotas)."
        )
    if cvm_facts:
        linhas.append("\nFatos Relevantes CVM recentes:")
        linhas.append(cvm_facts)
        
    return "\n".join(linhas)

def get_system_prompt(context: str) -> str:
    """
    Retorna o prompt do sistema para a IA.
    """
    return (
        "Você é um assistente que ajuda o usuário a entender o ranking de "
        "Fundos Imobiliários (FIIs) gerado pelo app dele. Responda em português, "
        "de forma clara e objetiva, com base SOMENTE nos dados fornecidos abaixo. "
        "Se a pergunta não puder ser respondida com esses dados, diga isso. "
        "⚠️ Nunca dê recomendação de compra/venda como certeza — deixe claro que é "
        "uma ferramenta de apoio, não uma recomendação financeira ou conselho profissional.\n\n"
        f"{context}"
    )

def stream_response(provider, client, messages, system_prompt):
    """
    Faz streaming da resposta da IA.
    """
    if provider == 'gemini':
        # format messages for genai
        # model = config.CHAT_MODEL_GEMINI
        gemini_messages = [{"role": m["role"], "parts": [{"text": m["content"]}]} for m in messages]
        try:
            response = client.models.generate_content_stream(
                model=getattr(config, 'CHAT_MODEL_GEMINI', 'gemini-2.5-flash'),
                contents=gemini_messages,
                config={"system_instruction": system_prompt}
            )
            for chunk in response:
                yield chunk.text
        except Exception as e:
            logger.error(f"Erro Gemini: {e}")
            yield f"Erro na comunicação com a IA: {e}"
            
    elif provider == 'claude':
        # format messages for anthropic
        anthropic_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        try:
            with client.messages.stream(
                model=getattr(config, 'CHAT_MODEL_CLAUDE', 'claude-3-5-sonnet-20241022'),
                max_tokens=getattr(config, 'CHAT_MAX_TOKENS', 1200),
                system=system_prompt,
                messages=anthropic_messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Erro Claude: {e}")
            yield f"Erro na comunicação com a IA: {e}"

def check_rate_limit(session_state) -> tuple:
    """
    Verifica se o limite de mensagens foi excedido na sessão.
    """
    max_msgs = getattr(config, 'CHAT_MAX_MSGS_PER_SESSION', 30)
    history = session_state.get('chat_history', [])
    user_msgs = len([m for m in history if m['role'] == 'user'])
    
    if user_msgs >= max_msgs:
        return False, f"Limite de {max_msgs} mensagens por sessão atingido."
    return True, ""

def summarize_document(provider, client, text: str, max_bullets=3) -> str:
    """
    Resume um documento ou fato relevante CVM em bullet points.
    """
    prompt = f"Resuma o texto abaixo em no máximo {max_bullets} bullet points concisos:\n\n{text}"
    
    if provider == 'gemini':
        try:
            response = client.models.generate_content(
                model=getattr(config, 'CHAT_MODEL_GEMINI', 'gemini-2.5-flash'),
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Erro ao resumir: {e}"
            
    elif provider == 'claude':
        try:
            response = client.messages.create(
                model=getattr(config, 'CHAT_MODEL_CLAUDE', 'claude-3-5-sonnet-20241022'),
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return "".join(block.text for block in response.content if block.type == "text")
        except Exception as e:
            return f"Erro ao resumir: {e}"
            
    return "Nenhum provedor de IA disponível para resumir."
