import streamlit as st
import pymongo
import google.generativeai as genai
from PyPDF2 import PdfReader
import base64
import os
import cohere
import time
# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = "AIzaSyAghyegpsWKXD_bu4o6N-BKzGgl_A0U6po"
MONGODB_URI = "mongodb+srv://savincesl_db_user:sebas1,Leon@csavl.xvmr26i.mongodb.net/"
COHERE_API_KEY = "ygQJd23Cpd5n4NvVIorSjAkdcqnNv9lAWOSZUHVu"

USER = "Sebastian Alonso Vinces Leon"

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan las variables de entorno GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
co = cohere.Client(COHERE_API_KEY)

# Conexión a MongoDB Atlas
client = pymongo.MongoClient(MONGODB_URI)
db = client["pdf_embeddings_db"]
collection = db["pdf_vectors"]

# =======================
# FUNCIONES
# =======================
def crear_indice_vectorial():
  from pymongo.operations import SearchIndexModel

  # Conexión a MongoDB Atlas
  client = pymongo.MongoClient(MONGODB_URI)
  db = client.pdf_embeddings_db
  collection = db.pdf_vectors
  collection.insert_one({"a":"sample"})

  existing_indexes = [index['name'] for index in collection.list_search_indexes()]
  if "vector_index" in existing_indexes:
    print("El índice 'vector_index' ya existe. No se crea nuevamente.")
    return

  # Create your index model, then create the search index
  search_index_model = SearchIndexModel(
    definition = {
      "fields": [
        {
          "type": "vector",
          "path": "embedding",
          "similarity": "cosine",
          "numDimensions": 768
        }
      ]
    },
    name="vector_index",
    type="vectorSearch"
  )

  collection.create_search_index(model=search_index_model)
  time.sleep(20)

crear_indice_vectorial()

def leer_pdf(archivo):
    reader = PdfReader(archivo)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() + "\n"
    return texto.strip()

def crear_embedding(texto):
    """Genera embeddings usando Cohere (modelo multilenguaje)."""
    resp = co.embed(
        model="multilingual-22-12",
        texts=[texto]
    )
    return resp.embeddings[0]

def procesar_pdf(archivo_pdf, nombre_pdf):
    """Lee PDF, genera embeddings, guarda en MongoDB y sube PDF a Backblaze."""
    st.info("📄 Leyendo PDF...")

    texto = leer_pdf(archivo_pdf)
    if not texto:
        st.error("El PDF no contiene texto.")
        return None

    trozos = [texto[i:i + 1000] for i in range(0, len(texto), 1000)]

    documentos = []
    for i, chunk in enumerate(trozos):
        embedding = crear_embedding(chunk)
        documentos.append({
            "pdf": nombre_pdf,
            "id": i,
            "texto": chunk,
            "embedding": embedding
        })

    # Guardar en MongoDB
    collection.insert_many(documentos)
    
    return len(documentos)

def buscar_similares(embedding, k=5):
    """
    Busca los documentos más similares en MongoDB Atlas.
    Requiere que el índice vectorial haya sido creado desde Atlas UI.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    return list(collection.aggregate(pipeline))

def generar_respuesta(pregunta, contextos):
    """Usa Gemini para responder con contexto"""
    modelo = genai.GenerativeModel("gemini-flash-latest")
    contexto = "\n\n".join([c["texto"] for c in contextos])
    prompt = f"""
Eres un asistente experto. Usa el siguiente contexto para responder la pregunta del usuario.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde de forma concisa y clara en español.
"""
    respuesta = modelo.generate_content(prompt)
    return respuesta.text

# =======================
# INTERFAZ STREAMLIT
# =======================

st.set_page_config(page_title="Chat PDF con MongoDB + API Gemini + API Cohere: ", page_icon="💬")
st.title("💬 Chat PDF con MongoDB + Gemini + Cohere: "+USER)

archivo_pdf = st.file_uploader("📤 Sube un PDF", type=["pdf"])

if archivo_pdf:
    if st.button("Procesar y guardar PDF"):
        with st.spinner("Procesando PDF..."):
            cantidad = procesar_pdf(archivo_pdf, archivo_pdf.name)
            st.success(f"Procesado: {cantidad} fragmentos generados y PDF guardado.")

st.subheader("💬 Pregunta sobre el contenido del PDF")

if "historial" not in st.session_state:
    st.session_state.historial = []

pregunta = st.chat_input("Escribe tu pregunta sobre el PDF...")

if pregunta:
    with st.spinner("Buscando respuesta..."):
        emb = crear_embedding(pregunta)
        similares = buscar_similares(emb, k=5)

        if not similares:
            respuesta = "No encontré informacion relevante en el documento."
        else:
            respuesta = generar_respuesta(pregunta, similares)

        st.session_state.historial.append({"rol": "usuario", "texto": pregunta})
        st.session_state.historial.append({"rol": "bot", "texto": respuesta})

# Mostrar historial
for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.chat_message("user").write(msg["texto"])
    else:
        st.chat_message("assistant").write(msg["texto"])


