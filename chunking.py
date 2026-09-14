import os
import time
import re

from langchain_community.document_loaders import FireCrawlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("FIRECRAWL_API_KEY")

if not api_key:
    print("API key not found!")
    exit()

os.environ["FIRECRAWL_API_KEY"] = api_key

# Tietokantaan lisättävät Memory Alpha -sivut
# HUOM! Testaa ensin pienellä määrällä, esim:
urls = [
    "https://memory-alpha.fandom.com/wiki/Jean-Luc_Picard",
    "https://memory-alpha.fandom.com/wiki/United_Federation_of_Planets",
    "https://memory-alpha.fandom.com/wiki/Warp_drive",
    "https://memory-alpha.fandom.com/wiki/USS_Enterprise_(NCC-1701-D)",
    "https://memory-alpha.fandom.com/wiki/Borg"
]

documents = []
print("Loading articles with Firecrawl...")

for url in urls:
    print(f"Fetching: {url}")
    loader = FireCrawlLoader(url=url, mode="scrape")
    docs = loader.load()
    documents.extend(docs)
    
    time.sleep(2)

print(f"\nLoaded {len(documents)} articles successfully.")

print("Cleaning up articles with regex...")

for doc in documents:
    text = doc.page_content
    
    # 1. Poistetaan sisällysluettelot (Table of Contents)
    # Memory Alphassa (ja Fandomissa yleensä) sisällysluettelo on omalla rivillään 
    # oleva sana "Contents" (tai otsikko), jota seuraa joukko listakohtia.
    # Tämä lauseke etsii sanan "Contents" ja tuhoaa kaikki sen jälkeiset rivit, 
    # jotka alkavat listamerkeillä (*, -, numerot, välilyönnit tai hakasulkeet)
    text = re.sub(r'(?im)^(#+\s+|\*\*?)?Contents(\*\*?)?\s*\n(?:^[\s\*\-\d\[].*\n)*', '', text)

    # 2. Poistetaan Markdown-linkeistä url-linkit, mutta jätetään teksti
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # 3. Poistetaan mahdolliset Wikipediasta/Fandomista tutut [edit]-tekstit
    text = text.replace("[edit]", "").replace("[ muokkaa ]", "")
    
    # 4. Siivotaan ylimääräiset tyhjät rivit, jotka vievät tilaa
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # 5. Siivotaan kursivoinnit (alaviivat) ja lihavoinnit (tähdet)
    text = text.replace("_", "").replace("**", "").replace("*", "")
    
    # Tallennetaan siivottu teksti takaisin dokumenttiin
    doc.page_content = text

print("Data cleaned!")

print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=400
)
chunks = text_splitter.split_documents(documents)
print(f"Split to {len(chunks)} chunks.")

print("\nLoading embedding model and saving to database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_free"
)
print("Done!")