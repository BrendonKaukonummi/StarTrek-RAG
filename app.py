import streamlit as st
import warnings
import os

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Jos käytetään Llama 3 -kielimallia:
# from langchain_community.llms import Ollama

# Jos käytetään OpenAI:n gpt-4o-mini -kielimallia (vaatii OpenAI:n maksullisen API-avaimen):
from langchain_openai import ChatOpenAI

# warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# 1. SIVUN ASETUKSET

st.set_page_config(page_title="Tähtilaivaston tietokone", layout="centered")
st.title("Tähtilaivaston tietokone")
st.caption("Kysy mitä tahansa Star Trek -sarjoista The Next Generation, Deep Space Nine ja Voyager. Tietokone etsii vastauksen tietokannasta ja kääntää sen suomeksi.")


# 2. RAG-TAUSTAJÄRJESTELMÄN LATAUS

@st.cache_resource(show_spinner=False)
def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory="./chroma_db_free", embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    # Jos käytetään Llama 3 -kielimallia:
    # llm = Ollama(model="llama3")

    # Jos käytetään OpenAI:n gpt-4o-mini -kielimallia (vaatii OpenAI:n maksullisen API-avaimen):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Historian ymmärtävä prompti (kääntää haun englanniksi)
    contextualize_q_system_prompt = (
        "Olet tekoäly, jonka tehtävänä on luoda hakulausekkeita englanninkieliseen tietokantaan. "
        "Ottaen huomioon chathistorian ja viimeisimmän käyttäjän kysymyksen, "
        "muotoile kysymys uudelleen itsenäiseksi, tarkaksi hakukysymykseksi. "
        "TÄRKEÄÄ: Käännä tämä hakukysymys AINA ENGLANNIKSI, koska tietokannan data on englanniksi. "
        "Tämä parantaa hakuosumia. ÄLÄ vastaa kysymykseen, vaan palauta pelkkä englanninkielinen hakulause."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Luodaan haku-ketju, joka on tietoinen historiasta
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Varsinainen vastaus-prompti (sisältää myös historian)
    qa_system_prompt = (
        "Olet Tähtilaivaston tietokoneen älykäyttöliittymä. Toimit konemaisesti ja ytimekkäästi.\n\n"
        "Jos annetussa kontekstissa ei ole tietoa, jonka avulla kysymykseen voi vastata, "
        "sinun on vastattava: 'Tietoa ei löydy tietokannasta.' Älä yritä päätellä, arvailla tai keksiä vastausta.\n\n"
        "Muut säännöt:\n"
        "- Vastaa samalla kielellä kuin käyttäjän kysymys.\n"
        "- Jos vastaat suomeksi, käännä Star Trek -termit sujuvasti suomeksi (esim. warp drive = poimuajo).\n"
        "- Älä käytä omaa ulkopuolista tietoasi vastauksen keksimiseen.\n"
        "- Älä koskaan tervehdi tai esittele itseäsi.\n"
        "- Älä koskaan keksi omia linkkejä.\n\n"
        "Konteksti:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    
    return rag_chain

with st.spinner("Ladataan tekoälymallia ja tietokantaa..."):
    chain = load_rag_chain()


# 3. CHAT-KÄYTTÖLIITTYMÄ

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Tervehdys. Olen Tähtilaivaston tietokone. Kuinka voin palvella?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 4. KÄYTTÄJÄN SYÖTE JA VASTAUS

if prompt := st.chat_input("Kysy tietokoneelta... (esim. Kuka oli tähtialus Enterprise-D:n kapteeni?)"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Käännetään Streamlitin historia LangChainin formaattiin
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        answer_text = ""
        docs_used = []
        
        # 2. Syötetään ketjulle 'input' ja 'chat_history'
        for chunk in chain.stream({"input": prompt, "chat_history": chat_history}):
            if 'answer' in chunk:
                answer_text += chunk['answer']
                message_placeholder.markdown(answer_text + "▌")
            
            if 'context' in chunk:
                docs_used = chunk['context']

        # DEBUG: Tulostetaan VS Coden terminaaliin ne palaset, jotka LLM oikeasti näkee
            # if 'context' in chunk:
            #     docs_used = chunk['context']
            #     print("\n=== DEBUG: TEKOÄLYLLE LÄHETETYT PALASET ===")
            #     for i, doc in enumerate(docs_used):
            #         print(f"Pala {i+1}: {doc.page_content[:350]}...")
            #     print("===========================================\n")
        
        # 3. Kun striimaus on valmis, lisätään lähteet VAIN, jos tieto löytyi
        if "tietoa ei löydy" in answer_text.lower() or "information not found" in answer_text.lower():
            final_answer = answer_text
        else:
            sources_text = "\n\n---\n**Käytetyt lähteet (Memory Alpha):**\n"
            unique_sources = set()
            
            for doc in docs_used:
                meta = doc.metadata
                source = meta.get('source') or meta.get('url') or meta.get('sourceURL') or "Tuntematon lähde"
                unique_sources.add(source)
                
            for source in unique_sources:
                sources_text += f"- {source}\n"
                
            final_answer = answer_text + sources_text
        
        # Päivitetään ruutu viimeisen kerran (poistetaan vilkkuva kursori ja lisätään lähteet)
        message_placeholder.markdown(final_answer)
            
    # 4. Tallennetaan koko vastaus historiaan
    st.session_state.messages.append({"role": "assistant", "content": final_answer})