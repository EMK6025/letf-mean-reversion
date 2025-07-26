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

def connect_time_series(engine, table_name = "test_data"):
    df = connect(engine, table_name)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

# use with wfo_run, wfo_period_summary, or wfo_strategy
def connect(engine, table_name = "wfo_run"):
    df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
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

