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

# Memory Alpha pages to add to the database:

urls = [

    "https://memory-alpha.fandom.com/wiki/Star_Trek:_The_Next_Generation",
    "https://memory-alpha.fandom.com/wiki/USS_Enterprise_(NCC-1701-D)",

    # Main characters:

    "https://memory-alpha.fandom.com/wiki/Jean-Luc_Picard",
    "https://memory-alpha.fandom.com/wiki/William_T._Riker",
    "https://memory-alpha.fandom.com/wiki/Geordi_La_Forge",
    "https://memory-alpha.fandom.com/wiki/Natasha_Yar",
    "https://memory-alpha.fandom.com/wiki/Worf",
    "https://memory-alpha.fandom.com/wiki/Beverly_Crusher",
    "https://memory-alpha.fandom.com/wiki/Deanna_Troi",
    "https://memory-alpha.fandom.com/wiki/Data",
    "https://memory-alpha.fandom.com/wiki/Wesley_Crusher",

    # Recurring characters:

    "https://memory-alpha.fandom.com/wiki/Guinan",
    "https://memory-alpha.fandom.com/wiki/Miles_O%27Brien",
    "https://memory-alpha.fandom.com/wiki/Keiko_O%27Brien",
    "https://memory-alpha.fandom.com/wiki/Gowron",
    "https://memory-alpha.fandom.com/wiki/Kurn",
    "https://memory-alpha.fandom.com/wiki/Lursa",
    "https://memory-alpha.fandom.com/wiki/B%27Etor",
    "https://memory-alpha.fandom.com/wiki/Alexander_Rozhenko",
    "https://memory-alpha.fandom.com/wiki/Lwaxana_Troi",
    "https://memory-alpha.fandom.com/wiki/Q",
    "https://memory-alpha.fandom.com/wiki/Thomas_Riker",
    "https://memory-alpha.fandom.com/wiki/Lore",
    "https://memory-alpha.fandom.com/wiki/Katherine_Pulaski",
    "https://memory-alpha.fandom.com/wiki/Reginald_Barclay",
    "https://memory-alpha.fandom.com/wiki/Vash",
    "https://memory-alpha.fandom.com/wiki/Sela",
    "https://memory-alpha.fandom.com/wiki/Ro_Laren",
    "https://memory-alpha.fandom.com/wiki/Locutus_of_Borg",
    "https://memory-alpha.fandom.com/wiki/Hugh",
    "https://memory-alpha.fandom.com/wiki/James_Moriarty_(hologram)",

    # Species:

    "https://memory-alpha.fandom.com/wiki/Human",
    "https://memory-alpha.fandom.com/wiki/Borg",
    "https://memory-alpha.fandom.com/wiki/Klingon",
    "https://memory-alpha.fandom.com/wiki/Romulan",
    "https://memory-alpha.fandom.com/wiki/Ferengi",
    "https://memory-alpha.fandom.com/wiki/Vulcan",
    "https://memory-alpha.fandom.com/wiki/Cardassian",
    "https://memory-alpha.fandom.com/wiki/Betazoid",
    "https://memory-alpha.fandom.com/wiki/El-Aurian",
    "https://memory-alpha.fandom.com/wiki/Bynar",
    "https://memory-alpha.fandom.com/wiki/Bajoran",
    "https://memory-alpha.fandom.com/wiki/The_Children_of_Tama",
    "https://memory-alpha.fandom.com/wiki/J%27naii",

    # Planets:

    "https://memory-alpha.fandom.com/wiki/Earth",
    "https://memory-alpha.fandom.com/wiki/Qo%27noS",
    "https://memory-alpha.fandom.com/wiki/Ni%27Var",
    "https://memory-alpha.fandom.com/wiki/Romulus",
    "https://memory-alpha.fandom.com/wiki/Betazed",
    "https://memory-alpha.fandom.com/wiki/Risa",

    # Other:

    "https://memory-alpha.fandom.com/wiki/Gene_Roddenberry",
    "https://memory-alpha.fandom.com/wiki/Library_Computer_Access_and_Retrieval_System",
    "https://memory-alpha.fandom.com/wiki/Personal_Access_Display_Device",
    "https://memory-alpha.fandom.com/wiki/Replicator",
    "https://memory-alpha.fandom.com/wiki/Earl_Grey_tea",
    "https://memory-alpha.fandom.com/wiki/Holodeck",
    "https://memory-alpha.fandom.com/wiki/Warp_factor",
    "https://memory-alpha.fandom.com/wiki/Warp_drive",
    "https://memory-alpha.fandom.com/wiki/Engineering",
    "https://memory-alpha.fandom.com/wiki/Sickbay",
    "https://memory-alpha.fandom.com/wiki/Bridge",
    "https://memory-alpha.fandom.com/wiki/Ten_Forward",
    "https://memory-alpha.fandom.com/wiki/Combadge",
    "https://memory-alpha.fandom.com/wiki/Transporter",
    "https://memory-alpha.fandom.com/wiki/Assimilation",
    "https://memory-alpha.fandom.com/wiki/Borg_cube",

    # EPISODES:

    # Season 1:

    "https://memory-alpha.fandom.com/wiki/Encounter_at_Farpoint_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Naked_Now_(episode)",
    "https://memory-alpha.fandom.com/wiki/Code_of_Honor_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Last_Outpost_(episode)",
    "https://memory-alpha.fandom.com/wiki/Where_No_One_Has_Gone_Before_(episode)",
    "https://memory-alpha.fandom.com/wiki/Lonely_Among_Us_(episode)",
    "https://memory-alpha.fandom.com/wiki/Justice_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Battle_(episode)",
    "https://memory-alpha.fandom.com/wiki/Hide_And_Q_(episode)",
    "https://memory-alpha.fandom.com/wiki/Haven_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Big_Goodbye_(episode)",
    "https://memory-alpha.fandom.com/wiki/Datalore_(episode)",
    "https://memory-alpha.fandom.com/wiki/Angel_One_(episode)",
    "https://memory-alpha.fandom.com/wiki/11001001_(episode)",
    "https://memory-alpha.fandom.com/wiki/Too_Short_A_Season_(episode)",
    "https://memory-alpha.fandom.com/wiki/When_The_Bough_Breaks_(episode)",
    "https://memory-alpha.fandom.com/wiki/Home_Soil_(episode)",
    "https://memory-alpha.fandom.com/wiki/Coming_of_Age_(episode)",
    "https://memory-alpha.fandom.com/wiki/Heart_of_Glory_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Arsenal_of_Freedom_(episode)",
    "https://memory-alpha.fandom.com/wiki/Symbiosis_(episode)",
    "https://memory-alpha.fandom.com/wiki/Skin_Of_Evil_(episode)",
    "https://memory-alpha.fandom.com/wiki/We%27ll_Always_Have_Paris_(episode)",
    "https://memory-alpha.fandom.com/wiki/Conspiracy_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Neutral_Zone_(episode)",

    # Season 2:

    "https://memory-alpha.fandom.com/wiki/The_Child_(episode)",
    "https://memory-alpha.fandom.com/wiki/Where_Silence_Has_Lease_(episode)",
    "https://memory-alpha.fandom.com/wiki/Elementary,_Dear_Data_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Outrageous_Okona_(episode)",
    "https://memory-alpha.fandom.com/wiki/Loud_As_A_Whisper_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Schizoid_Man_(episode)",
    "https://memory-alpha.fandom.com/wiki/Unnatural_Selection_(episode)",
    "https://memory-alpha.fandom.com/wiki/A_Matter_Of_Honor_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Measure_Of_A_Man_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Dauphin_(episode)",
    "https://memory-alpha.fandom.com/wiki/Contagion_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Royale_(episode)",
    "https://memory-alpha.fandom.com/wiki/Time_Squared_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Icarus_Factor_(episode)",
    "https://memory-alpha.fandom.com/wiki/Pen_Pals_(episode)",
    "https://memory-alpha.fandom.com/wiki/Q_Who_(episode)",
    "https://memory-alpha.fandom.com/wiki/Samaritan_Snare_(episode)",
    "https://memory-alpha.fandom.com/wiki/Up_The_Long_Ladder_(episode)",
    "https://memory-alpha.fandom.com/wiki/Manhunt_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Emissary_(episode)",
    "https://memory-alpha.fandom.com/wiki/Peak_Performance_(episode)",
    "https://memory-alpha.fandom.com/wiki/Shades_of_Gray_(episode)",

    # Season 3:

    "https://memory-alpha.fandom.com/wiki/Evolution_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Ensigns_of_Command_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Survivors_(episode)",
    "https://memory-alpha.fandom.com/wiki/Who_Watches_The_Watchers_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Bonding_(episode)",
    "https://memory-alpha.fandom.com/wiki/Booby_Trap_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Enemy_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Price_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Vengeance_Factor_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Defector_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Hunted_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_High_Ground_(episode)",
    "https://memory-alpha.fandom.com/wiki/Deja_Q_(episode)",
    "https://memory-alpha.fandom.com/wiki/A_Matter_of_Perspective_(episode)",
    "https://memory-alpha.fandom.com/wiki/Yesterday%27s_Enterprise_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Offspring_(episode)",
    "https://memory-alpha.fandom.com/wiki/Sins_of_The_Father_(episode)",
    "https://memory-alpha.fandom.com/wiki/Allegiance_(episode)",
    "https://memory-alpha.fandom.com/wiki/Captain%27s_Holiday_(episode)",
    "https://memory-alpha.fandom.com/wiki/Tin_Man_(episode)",
    "https://memory-alpha.fandom.com/wiki/Hollow_Pursuits_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Most_Toys_(episode)",
    "https://memory-alpha.fandom.com/wiki/Sarek_(episode)",
    "https://memory-alpha.fandom.com/wiki/M%C3%A9nage_%C3%A0_Troi_(episode)",
    "https://memory-alpha.fandom.com/wiki/Transfigurations_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Best_of_Both_Worlds_(episode)",

    # Season 4:

    "https://memory-alpha.fandom.com/wiki/The_Best_of_Both_Worlds,_Part_II_(episode)",
    "https://memory-alpha.fandom.com/wiki/Family_(episode)",
    "https://memory-alpha.fandom.com/wiki/Brothers_(episode)",
    "https://memory-alpha.fandom.com/wiki/Suddenly_Human_(episode)",
    "https://memory-alpha.fandom.com/wiki/Remember_Me_(episode)",
    "https://memory-alpha.fandom.com/wiki/Legacy_(episode)",
    "https://memory-alpha.fandom.com/wiki/Reunion_(episode)",
    "https://memory-alpha.fandom.com/wiki/Future_Imperfect_(episode)",
    "https://memory-alpha.fandom.com/wiki/Final_Mission_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Loss_(episode)",
    "https://memory-alpha.fandom.com/wiki/Data%27s_Day_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Wounded_(episode)",
    "https://memory-alpha.fandom.com/wiki/Devil%27s_Due_(episode)",
    "https://memory-alpha.fandom.com/wiki/Clues_(episode)",
    "https://memory-alpha.fandom.com/wiki/First_Contact_(episode)",
    "https://memory-alpha.fandom.com/wiki/Galaxy%27s_Child_(episode)",
    "https://memory-alpha.fandom.com/wiki/Night_Terrors_(episode)",
    "https://memory-alpha.fandom.com/wiki/Identity_Crisis_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Nth_Degree_(episode)",
    "https://memory-alpha.fandom.com/wiki/Qpid_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Drumhead_(episode)",
    "https://memory-alpha.fandom.com/wiki/Half_a_Life_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Host_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Mind%27s_Eye_(episode)",
    "https://memory-alpha.fandom.com/wiki/In_Theory_(episode)",
    "https://memory-alpha.fandom.com/wiki/Redemption_(episode)",

    # Season 5:

    "https://memory-alpha.fandom.com/wiki/Redemption_II_(episode)",
    "https://memory-alpha.fandom.com/wiki/Darmok_(episode)",
    "https://memory-alpha.fandom.com/wiki/Ensign_Ro_(episode)",
    "https://memory-alpha.fandom.com/wiki/Silicon_Avatar_(episode)",
    "https://memory-alpha.fandom.com/wiki/Disaster_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Game_(episode)",
    "https://memory-alpha.fandom.com/wiki/Unification_I_(episode)",
    "https://memory-alpha.fandom.com/wiki/Unification_II_(episode)",
    "https://memory-alpha.fandom.com/wiki/A_Matter_Of_Time_(episode)",
    "https://memory-alpha.fandom.com/wiki/New_Ground_(episode)",
    "https://memory-alpha.fandom.com/wiki/Hero_Worship_(episode)",
    "https://memory-alpha.fandom.com/wiki/Violations_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Masterpiece_Society_(episode)",
    "https://memory-alpha.fandom.com/wiki/Conundrum_(episode)",
    "https://memory-alpha.fandom.com/wiki/Power_Play_(episode)",
    "https://memory-alpha.fandom.com/wiki/Ethics_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Outcast_(episode)",
    "https://memory-alpha.fandom.com/wiki/Cause_And_Effect_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_First_Duty_(episode)",
    "https://memory-alpha.fandom.com/wiki/Cost_Of_Living_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Perfect_Mate_(episode)",
    "https://memory-alpha.fandom.com/wiki/Imaginary_Friend_(episode)",
    "https://memory-alpha.fandom.com/wiki/I_Borg_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Next_Phase_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Inner_Light_(episode)",
    "https://memory-alpha.fandom.com/wiki/Time%27s_Arrow_(episode)",

    # Season 6:

    "https://memory-alpha.fandom.com/wiki/Time%27s_Arrow,_Part_II_(episode)",
    "https://memory-alpha.fandom.com/wiki/Realm_Of_Fear_(episode)",
    "https://memory-alpha.fandom.com/wiki/Man_Of_The_People_(episode)",
    "https://memory-alpha.fandom.com/wiki/Relics_(episode)",
    "https://memory-alpha.fandom.com/wiki/Schisms_(episode)",
    "https://memory-alpha.fandom.com/wiki/True_Q_(episode)",
    "https://memory-alpha.fandom.com/wiki/Rascals_(episode)",
    "https://memory-alpha.fandom.com/wiki/A_Fistful_of_Datas_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Quality_of_Life_(episode)",
    "https://memory-alpha.fandom.com/wiki/Chain_Of_Command,_Part_I_(episode)",
    "https://memory-alpha.fandom.com/wiki/Chain_Of_Command,_Part_II_(episode)",
    "https://memory-alpha.fandom.com/wiki/Ship_In_A_Bottle_(episode)",
    "https://memory-alpha.fandom.com/wiki/Aquiel_(episode)",
    "https://memory-alpha.fandom.com/wiki/Face_Of_The_Enemy_(episode)",
    "https://memory-alpha.fandom.com/wiki/Tapestry_(episode)",
    "https://memory-alpha.fandom.com/wiki/Birthright,_Part_I_(episode)",
    "https://memory-alpha.fandom.com/wiki/Birthright,_Part_II_(episode)",
    "https://memory-alpha.fandom.com/wiki/Starship_Mine_(episode)",
    "https://memory-alpha.fandom.com/wiki/Lessons_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Chase_(episode)",
    "https://memory-alpha.fandom.com/wiki/Frame_of_Mind_(episode)",
    "https://memory-alpha.fandom.com/wiki/Suspicions_(episode)",
    "https://memory-alpha.fandom.com/wiki/Rightful_Heir_(episode)",
    "https://memory-alpha.fandom.com/wiki/Second_Chances_(episode)",
    "https://memory-alpha.fandom.com/wiki/Timescape_(episode)",
    "https://memory-alpha.fandom.com/wiki/Descent_(episode)",

    # Season 7:

    "https://memory-alpha.fandom.com/wiki/Descent,_Part_II_(episode)",
    "https://memory-alpha.fandom.com/wiki/Liaisons_(episode)",
    "https://memory-alpha.fandom.com/wiki/Interface_(episode)",
    "https://memory-alpha.fandom.com/wiki/Gambit,_Part_I_(episode)",
    "https://memory-alpha.fandom.com/wiki/Gambit,_Part_II_(episode)",
    "https://memory-alpha.fandom.com/wiki/Phantasms_(episode)",
    "https://memory-alpha.fandom.com/wiki/Dark_Page_(episode)",
    "https://memory-alpha.fandom.com/wiki/Attached_(episode)",
    "https://memory-alpha.fandom.com/wiki/Force_of_Nature_(episode)",
    "https://memory-alpha.fandom.com/wiki/Inheritance_(episode)",
    "https://memory-alpha.fandom.com/wiki/Parallels_(episode)",
    "https://memory-alpha.fandom.com/wiki/The_Pegasus_(episode)",
    "https://memory-alpha.fandom.com/wiki/Homeward_(episode)",
    "https://memory-alpha.fandom.com/wiki/Sub_Rosa_(episode)",
    "https://memory-alpha.fandom.com/wiki/Lower_Decks_(episode)",
    "https://memory-alpha.fandom.com/wiki/Thine_Own_Self_(episode)",
    "https://memory-alpha.fandom.com/wiki/Masks_(episode)",
    "https://memory-alpha.fandom.com/wiki/Eye_of_the_Beholder_(episode)",
    "https://memory-alpha.fandom.com/wiki/Genesis_(episode)",
    "https://memory-alpha.fandom.com/wiki/Journey%27s_End_(episode)",
    "https://memory-alpha.fandom.com/wiki/Firstborn_(episode)",
    "https://memory-alpha.fandom.com/wiki/Bloodlines_(episode)",
    "https://memory-alpha.fandom.com/wiki/Emergence_(episode)",
    "https://memory-alpha.fandom.com/wiki/Preemptive_Strike_(episode)",
    "https://memory-alpha.fandom.com/wiki/All_Good_Things..._(episode)"

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
    
    # 1. Poistetaan sisällysluettelo (Table of Contents) turvallisesti
    text = re.sub(r'(?im)^(#+\s*)?\**Contents\**\s*\n(?:^\s*[\*\-\d]+\.?\s*\[.*?\n)+', '', text)

    # 2. Poistetaan Markdown-linkeistä url-linkit, mutta jätetään teksti
    text = re.sub(r'\[([^\]]+)\]\((?:[^)(]+|\([^)(]*\))*\)', r'\1', text)
    
    # 3. Poistetaan mahdolliset Wikipediasta/Fandomista tutut [edit]-tekstit ja muut turhat tekstit
    text = text.replace("[edit]", "").replace("!toggle section", "").replace("REPORT ISSUE", "").replace("Provided by: Fandom", "")

    # 4. Poistetaan "Sign in to edit" -tekstit ja niiden kenoviivat
    text = re.sub(r'\\?\s*\[Sign in to edit\\?\]', '', text, flags=re.IGNORECASE)
    
    # 5. Siivotaan ylimääräiset tyhjät rivit, jotka vievät tilaa
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # 6. Siivotaan kursivoinnit (alaviivat) ja lihavoinnit (tähdet)
    text = text.replace("_", "").replace("**", "").replace("*", "")

    # 7. Poistetaan kuvat ja niiden perässä olevat kuvatekstit
    text = re.sub(r'!\[(.*?)\]\((?:[^)(]+|\([^)(]*\))*\)(?:\s*\n+\s*\1)?', '', text)
    
    # Tallennetaan siivottu teksti takaisin dokumenttiin
    doc.page_content = text

print("Data cleaned!")

print("Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
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