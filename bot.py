import logging
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import traceback

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

TELEGRAM_BOT_TOKEN = "604146692:AAG1eYAo53H7RCMEd50TW6F-D8ujI73Qvfc"
FASTAPI_URL = "http://localhost:8000"


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Welcome handler framing bot operational parameters."""
    await update.message.reply_text(
        "👋 Welcome! I am your local Intel Mac RAG Assistant.\n\n"
        "📁 **Ingestion:** Send me any PDF or Word document (.docx), and I will process it into Qdrant.\n"
        "💬 **Retrieval:** Ask me any regular text question, and I will query the knowledge base using local LLMs."
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Intercepts files and pushes payloads through the FastAPI ingestion network."""
    doc = update.message.document
    filename = doc.file_name

    if not (filename.endswith('.pdf') or filename.endswith('.docx')):
        await update.message.reply_text("❌ Please send only valid document files ending in .pdf or .docx.")
        return

    status_msg = await update.message.reply_text(f"📥 Downloading and parsing {filename} locally...")

    # Download the document block from Telegram cloud servers
    telegram_file = await context.bot.get_file(doc.file_id)
    file_bytes = await telegram_file.download_as_bytearray()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Prepare standard multipart form files structure mapping
        files = {'file': (filename, bytes(file_bytes), doc.mime_type)}
        data = {'file_id': doc.file_id}

        try:
            response = await client.post(f"{FASTAPI_URL}/ingest", files=files, data=data)
            if response.status_code == 200:
                res_data = response.json()
                await status_msg.edit_text(
                    f"✅ Successfully indexed {filename} into Qdrant ({res_data['chunks_indexed']} chunks).")
            else:
                await status_msg.edit_text(f"❌ Ingestion failed: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            await status_msg.edit_text(f"❌ Connection to backend API timed out or failed: {str(e)}")


async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Routes user text queries straight into the vector-generation loop with deep networking diagnostics."""
    user_question = update.message.text
    user_id = update.effective_user.id

    logging.info(f"======== 📥 NEW USER QUERY [User ID: {user_id}] ========")
    logging.info(f"User Question Text: '{user_question}'")

    # Send temporary placeholder status indicator to the user
    placeholder = await update.message.reply_text("🤔 Searching knowledge base and generating answer...")

    # Target URL we are trying to hit
    target_url = f"{FASTAPI_URL}/query"
    logging.info(f"[Step 1/4] Attempting HTTP connection to FastAPI server at target: {target_url}")

    async with httpx.AsyncClient(timeout=150.0) as client:
        try:
            # Send payload configuration
            json_payload = {"question": user_question}
            logging.info(f"[Step 2/4] Network socket open. Dispatching POST request payload: {json_payload}")

            response = await client.post(target_url, json=json_payload)

            logging.info(f"[Step 3/4] Response received from FastAPI server. HTTP Status Code: {response.status_code}")

            if response.status_code == 200:
                answer = response.json().get("answer", "")
                logging.info(f"[Step 4/4] Successfully parsed JSON. Answer character length: {len(answer)}")
                logging.info(f"Answer snippet: {answer[:100]}...")

                await placeholder.edit_text(answer)
                logging.info("======== 🏁 QUERY PIPELINE SUCCESS ========")
            else:
                error_detail = "Unknown internal server issue"
                try:
                    error_detail = response.json().get("detail", error_detail)
                except Exception:
                    error_detail = response.text

                logging.error(f"❌ [Step 4/4 Failure] FastAPI returned non-200 status. Details: {error_detail}")
                await placeholder.edit_text(
                    f"❌ The generation engine returned an inner exception processing that query: {error_detail}")

        except httpx.ConnectError as ce:
            logging.error("❌ CRITICAL: Connection Refused! Is app.py actually running on port 8000?")
            logging.error(f"Technical Exception details: {str(ce)}")
            await placeholder.edit_text(
                f"❌ API engine is unreachable: Connection refused by host on port 8000. Verify app.py is alive.")

        except httpx.TimeoutException as te:
            logging.error(
                "❌ CRITICAL: Request Timed Out! Your Mac CPU took longer than 90 seconds to compute the embedding/response.")
            logging.error(f"Technical Exception details: {str(te)}")
            await placeholder.edit_text(
                "❌ API engine is unreachable: The request timed out while generating your answer.")

        except Exception as e:
            logging.error("❌ CRITICAL: Unexpected network exception encountered!")
            logging.error(f"Exception Type: {type(e).__name__}")
            logging.error(f"Exception Message: {str(e)}")
            logging.error(f"Full Stack Trace:\n{traceback.format_exc()}")
            await placeholder.edit_text(f"❌ API engine is unreachable: {type(e).__name__} - {str(e)}")

        logging.info("==================================================\n")

def main():
    """Initializes the structural Telegram client pipeline loop wrapper."""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))

    # FIX: Instantiate the Document filter by adding parentheses ()
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Alternatively, if you want to be safe and catch all document types:
    # app.add_handler(MessageHandler(filters.ATTACHMENT, handle_document))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    logging.info("Telegram interface engine actively polling for inputs...")
    app.run_polling()


if __name__ == "__main__":
    main()
