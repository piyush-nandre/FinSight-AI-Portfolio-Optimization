from google import genai
import streamlit as st


client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


def generate_portfolio_insight(
    question,
    retrieved_chunks,
    selected_assets,
    risk_profile
):
    """
    Generate AI insight using
    retrieved portfolio news context.
    """

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are FinSight AI.

Analyze the user's portfolio using the provided news context.

Selected Assets:
{selected_assets}

Risk Profile:
{risk_profile}

User Question:
{question}

Retrieved News Context:
{context}

Instructions:

1. Answer the question.
2. Mention potential risks.
3. Mention potential opportunities.
4. Keep the response concise.
5. Use bullet points where useful.
6. Focus primarily on Indian markets.
7. Use the retrieved news context whenever relevant.

Response:
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            f"Gemini Error: {str(e)}"
        )


def generate_general_response(
    question
):
    """
    General-purpose Gemini response
    when no relevant financial news
    is found in FAISS retrieval.
    """

    prompt = f"""
You are FinSight AI.

The user has asked a question.

Question:
{question}

Instructions:

1. Answer directly.
2. Be concise.
3. Use bullet points where useful.
4. If the topic is finance-related, provide financial context.
5. If the topic is unrelated to finance, answer normally.

Response:
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            f"Gemini Error: {str(e)}"
        )