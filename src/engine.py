from sqlalchemy import create_engine

def connect():
    return create_engine("postgresql+psycopg://postgres:Mhkuu1.1213704313@localhost:5432/letf_data")