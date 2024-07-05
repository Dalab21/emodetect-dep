from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from database import Base


class Role(Base):
    __tablename__ = "role"
    role_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50))

class User(Base):
    __tablename__ = "user"
    id_user = Column(Integer, primary_key=True, index=True)
    username = Column(String(250), unique=True, index=True)
    email = Column(String(350), unique=True, index=True)
    password = Column(String(650))
    role_id = Column(Integer, ForeignKey("role.role_id"))

    role = relationship("Role")

