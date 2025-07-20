from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(Integer, unique=True, index=True, nullable=False)
    is_active = Column(Boolean, default=True)

    images = relationship("Images", back_populates="user")

class Images(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id")
    phone_number = Column(Integer, nullable=False)
    real_image = Column(LargeBinary, nullable=False)
    heatmap_gen = Column(LargeBinary, nullable=False)
    heatmap_dis = Column(LargeBinary, nullable=False)

    user = relationship("User", back_populates="images")
