### Setup

To get started, please install the following python libraries: 

```
vectorbt==0.27.3
deap==1.4.3
yfinance==0.2.61
SQLAlchemy==2.0.41
pandas==2.2.3
```

Also make sure you have PostgreSQL downloaded and installed, with sign-in credentials initialized in `engine.py`, which you have to create.

```py
from sqlalchemy import create_engine as _sa_create_engine

import pandas as pd

username = "username"
password = "password"

def create_engine():
    return _sa_create_engine(f"postgresql+psycopg://{username}:{password}@localhost:5432/letf_data")

def connect(engine, table_name = "test_data"):
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)
```

In a separate cmd line tab, start up Postgress if needed:
```bash
net start postgresql-x64-17
```

Then open a new terminal and run:
```bash
psql -U your_username -d postgres

CREATE DATABASE letf_data;
```

To be sure that this repo works properly, it is best to be on python 3.11.9 as vectorbt and deap are constantly being changed with various updates. Other python versions could work, but don't take my word for it. Best to use a virtual environment. 

Once you have your postgreSQL credentials initialized, run `python main.py` and the database should be populated. 

