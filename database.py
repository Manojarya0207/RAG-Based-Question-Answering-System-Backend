import os
import time
from sqlalchemy import Column, String, Float, Text, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./data/rag_system.db"

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DocumentModel(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String)
    status = Column(String) # pending, processing, ready, failed
    error = Column(String, nullable=True)
    created_at = Column(Float, default=time.time)
    summary = Column(Text, nullable=True) # For the new summary feature

class ChatMessageModel(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    role = Column(String) # user, assistant
    content = Column(Text)
    timestamp = Column(Float, default=time.time)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
