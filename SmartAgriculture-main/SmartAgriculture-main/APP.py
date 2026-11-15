        fetch_k=min(15, vectorstore.index.ntotal),
        lambda_mult=0.7,
    )

# -------------------------
# Generate answer with OpenRouter
# -------------------------
def generate_answer(question: str, docs):
    context_parts = []
    for i, doc in enumerate(docs, start=1):
        context_parts.append(
            f"[Document {i}]\n"
            f"Topic: {doc.metadata.get('topic', 'unknown')}\n"
            f"Content: {doc.page_content}\n"
            f"Source: {doc.metadata.get('source', 'Not specified')}\n"
            f"Reference: {doc.metadata.get('url', 'Not specified')}"
        )
    context = "\n\n".join(context_parts) if context_parts else "No relevant documents were retrieved."

    prompt = f"""
You are a helpful Smart Agriculture educational assistant.

Answer the user's agriculture-related question clearly and practically.
Use the retrieved knowledge-base documents when they are relevant.

Important rules:
1. First check whether the retrieved documents contain enough information.
2. If they do, base the answer on them and do not add unsupported claims.
3. If they do not contain enough information, you may still give a general educational
   answer using your general agricultural knowledge. Clearly add this note:
   "This part is general information and is not covered by the current knowledge base."
4. Never pretend that general model knowledge came from a retrieved source.
5. If the answer depends on crop, location, season, soil, or climate, explain that and
   ask for the missing details where useful.
6. Do not invent local recommendations, guaranteed yields, prices, or government schemes.
7. For pesticide/fertilizer use, do not invent exact chemical doses or mixing instructions.
   Explain that the product label and local agricultural extension expert should be followed.
8. Use simple language and headings/bullets when helpful.
9. Cite the source names from the retrieved documents when you use them.

Retrieved knowledge-base context:
{context}

User question:
{question}

Answer:
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are an accurate, cautious agriculture education assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except ValueError:
                detail = response.text
            return f"OpenRouter API error ({response.status_code}): {detail}"

        result = response.json()
        choices = result.get("choices", [])
        if not choices:
            return f"Unexpected OpenRouter response: {result}"
        return choices[0]["message"]["content"].strip()

    except requests.RequestException as exc:
        return f"Could not connect to OpenRouter: {exc}"
    except (KeyError, TypeError, ValueError) as exc:
        return f"Could not read the model response: {exc}"

# -------------------------
# Streamlit chat interface
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask any agriculture-related question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the agriculture knowledge base..."):
            docs = retrieve_documents(question)

        with st.spinner("Preparing the answer..."):
            answer = generate_answer(question, docs)

        st.markdown(answer)

        if docs:
            with st.expander("Retrieved documents and sources"):
                for doc in docs:
                    st.markdown(f"**Topic:** {doc.metadata.get('topic', 'unknown')}")
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Not specified')}")
                    url = doc.metadata.get("url")
                    if url:
                        st.markdown(f"[Open reference]({url})")
                    st.caption(doc.page_content)
                    st.divider()
        else:
            st.info("No documents were retrieved. The answer may rely on general model knowledge.")

    st.session_state.messages.append({"role": "assistant", "content": answer})
