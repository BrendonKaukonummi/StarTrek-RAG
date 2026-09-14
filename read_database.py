from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Tällä skriptillä voidaan lukea ja testata Chroma-tietokantaan tallennettuja palasia (chunks).
# Sen avulla varmistetaan, että teksti on luettavaa ja yksittäiset palaset ovat sopivan mittaisia.

# 1. Yhdistetään olemassa olevaan lokaaliin tietokantaan
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory="./chroma_db_free", embedding_function=embeddings)

# 2. Haetaan tietokannan sisältö
# .get() hakee oletuksena kaikki palaset (id:t, metadatan ja raakatekstin)
collection = vectorstore.get()

TOTAL_CHUNKS = len(collection['ids'])
print(f"Tietokannassa on yhteensä {TOTAL_CHUNKS} tekstinpalaa.\n")

# 3. Tulostetaan esim. 3 ensimmäistä palasta (jotta terminaali ei täyty koko tietokannasta)
chunks_to_print = min(3, TOTAL_CHUNKS)

for i in range(chunks_to_print):
    print(f"=== PALANEN {i+1} ===")
    print(f"LÄHDE: {collection['metadatas'][i]}")
    # Tulostetaan koko palasen teksti (chunk). 
    # Jos palanen on pitkä, voit lisätä [:300] nähdäksesi vain alun.
    print(f"TEKSTI:\n{collection['documents'][i]}") 
    print("=" * 50 + "\n")