### Setup

First, make sure you have PostgreSQL downloaded and installed. Afterwards, simply run `config.py` to setup and test this project. Or, if you're old school, continue by following the instructions below...

---

Install the following python libraries: 

```
vectorbt==0.27.3+
deap==1.4.3+
yfinance==0.2.61+
SQLAlchemy==2.0.41+
pandas==2.2.3+
```

Next, create `engine.py` using the boilerplate below and plug in your own username and password:

```py
from sqlalchemy import create_engine as _sa_create_engine

import pandas as pd

username = "username"
password = "password"

def create_engine():
    return _sa_create_engine(f"postgresql+psycopg://{username}:{password}@localhost:5432/letf_data")

def connect(engine, table_name = "test_data"):
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df
```

In a separate cmd line tab, start up Postgress if needed:

```bash
net start postgresql-x64-17
```

Then open a new terminal and run:
```bash
psql -U your_username

CREATE DATABASE letf_data;
```

To be sure that this repo works properly, it is best to be on python 3.11.9 as vectorbt and deap are constantly being changed with various updates. Other python versions could work, but don't take my word for it. Best to use a virtual environment. 

Once you have your postgreSQL credentials initialized, run `python main.py` and the database should be populated. 

It would also be best to run this in a virtual environment, but up to you. 

