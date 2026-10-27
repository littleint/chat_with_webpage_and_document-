import os
import shutil
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from sqlalchemy.orm import Session

from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import SimpleChatEngine

from app.loaders import load_web_url, load_pdf_file, load_txt_file
from app.models import Conversation, Message

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = BASE_DIR / "storage" / "indices"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


class RAGEngineManager:
    """Manages document vectorization, disk persistence, and per-conversation chat engines."""

    def __init__(self):
        # In-memory cache of loaded indices: {conv_id: VectorStoreIndex}
        self._indices_cache: Dict[str, VectorStoreIndex] = {}

    def _get_storage_path(self, conversation_id: str) -> Path:
        return STORAGE_DIR / conversation_id

    def has_persisted_index(self, conversation_id: str) -> bool:
        """Check if an index is persisted on disk for the given conversation."""
        path = self._get_storage_path(conversation_id)
        return path.exists() and (path / "docstore.json").exists()

    def get_or_load_index(self, conversation_id: str) -> Optional[VectorStoreIndex]:
        """Retrieve an index from memory cache or load it from disk."""
        if conversation_id in self._indices_cache:
            return self._indices_cache[conversation_id]

        storage_path = self._get_storage_path(conversation_id)
        if self.has_persisted_index(conversation_id):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
                index = load_index_from_storage(storage_context)
                self._indices_cache[conversation_id] = index
                return index
            except Exception as e:
                print(f"Error loading index for conversation {conversation_id}: {e}")
                return None
        return None

    def save_index(self, conversation_id: str, index: VectorStoreIndex):
        """Persist index to disk and update cache."""
        storage_path = self._get_storage_path(conversation_id)
        storage_path.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(storage_path))
        self._indices_cache[conversation_id] = index

    def index_url(self, conversation_id: str, url: str) -> Tuple[bool, str]:
        """Scrape and index a webpage URL for a conversation."""
        try:
            docs = load_web_url(url)
            print(f"Building vector index for conversation {conversation_id}: {url} ({len(docs)} chunks)...", flush=True)
            index = VectorStoreIndex.from_documents(docs)
            self.save_index(conversation_id, index)
            return True, f"Successfully indexed '{url}' ({len(docs)} sections loaded)."
        except Exception as e:
            return False, f"Failed to index URL: {str(e)}"

    def index_file(self, conversation_id: str, temp_file_path: str, filename: str) -> Tuple[bool, str]:
        """Load and index a PDF or TXT file for a conversation."""
        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".pdf":
                docs = load_pdf_file(temp_file_path, filename=filename)
            elif ext in [".txt", ".md", ".log", ".csv", ".json"]:
                docs = load_txt_file(temp_file_path, filename=filename)
            else:
                return False, f"Unsupported file type: {ext}. Please upload a PDF or TXT file."

            print(f"Building vector index for conversation {conversation_id}: {filename} ({len(docs)} chunks)...", flush=True)
            index = VectorStoreIndex.from_documents(docs)
            self.save_index(conversation_id, index)
            return True, f"Successfully indexed '{filename}' ({len(docs)} sections loaded)."
        except Exception as e:
            return False, f"Failed to index file: {str(e)}"

    def delete_conversation_index(self, conversation_id: str):
        """Clean up on-disk storage and cache for a deleted conversation."""
        self._indices_cache.pop(conversation_id, None)
        storage_path = self._get_storage_path(conversation_id)
        if storage_path.exists():
            try:
                shutil.rmtree(storage_path)
            except Exception as e:
                print(f"Warning: Failed to delete index folder {storage_path}: {e}")

    def chat(self, conversation_id: str, question: str, db: Session) -> str:
        """Query the conversation using its document index and message history."""
        # 1. Fetch past messages for this conversation from SQLite to re-populate memory
        past_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )

        chat_history = []
        for msg in past_messages:
            role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
            chat_history.append(ChatMessage(role=role, content=msg.content))

        memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history, token_limit=3500)

        # 2. Check if this conversation has an indexed document
        index = self.get_or_load_index(conversation_id)

        if index:
            # Document is attached: perform RAG with conversation memory
            chat_engine = index.as_chat_engine(
                chat_mode="condense_plus_context",
                llm=Settings.llm,
                memory=memory,
                verbose=False
            )
            response = chat_engine.chat(question)
            return str(response)
        else:
            # No document attached: general conversation mode using LLM
            chat_engine = SimpleChatEngine.from_defaults(llm=Settings.llm, memory=memory)
            response = chat_engine.chat(question)
            return str(response)

    def stream_chat(self, conversation_id: str, question: str, db: Session):
        """Query conversation and yield tokens as a real-time generator."""
        past_messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .all()
        )

        chat_history = []
        for msg in past_messages:
            role = MessageRole.USER if msg.role == "user" else MessageRole.ASSISTANT
            chat_history.append(ChatMessage(role=role, content=msg.content))

        memory = ChatMemoryBuffer.from_defaults(chat_history=chat_history, token_limit=3500)
        index = self.get_or_load_index(conversation_id)

        if index:
            chat_engine = index.as_chat_engine(
                chat_mode="condense_plus_context",
                llm=Settings.llm,
                memory=memory,
                verbose=False
            )
            streaming_response = chat_engine.stream_chat(question)
            return streaming_response.response_gen
        else:
            chat_engine = SimpleChatEngine.from_defaults(llm=Settings.llm, memory=memory)
            streaming_response = chat_engine.stream_chat(question)
            return streaming_response.response_gen


# Singleton manager instance
rag_engine = RAGEngineManager()
