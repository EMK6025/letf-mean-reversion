# Mean Reversion vs Buy-and-Hold in S&P500 LETFs

This repo is the custom GA backtesting pipline I developed as apart of my Summer 2025 Erickson Grant project, where I looked into the tradeoffs of using leveraged ETFs for short term mean reversion trades versus long term buy-and-hold strategies. Through an evolutionary approach, I aimed to verify the conventional view towards LETFs, that they're only suited for short-term trades and should be avoided for long-term investments. 

In a summary, the results find that the risk-optimal strategies stayed away from buy-and-hold, preferring a short-term swing trading strategy with a holding period of around a few months. When simply letting idle cash sit around, the strategy matches the benchmark at a beta < 1 due to the portfolio exposure, however if the cash is held in short-term treasuries instead (proxied with RF rates), statistically significant alpha is created. More details can be found in the reports folder, with both my submitted proposal and technical/non-technical reports. 

### Setup

Step 1: Download, install, and setup [PostgreSQL](https://www.postgresql.org/download/)

Once the download and installation are complete, start up Postgress from your terminal if necessary:

```bash
net start postgresql-x64-17
```

Then create the database:
```bash
psql -U your_username

CREATE DATABASE letf_data;
```
More instructions can be found on the PostgreSQL website. 

---

Step 2: Setting up python libraries

You can either  run the `config.py` script to setup and validate, or you can follow the instructions below:


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

username = 'username'
password = 'password'

def create_engine():
    return _sa_create_engine(f'postgresql+psycopg://{username}:{password}@localhost:5432/letf_data')

def connect_time_series(engine, table_name = 'test_data'):
    df = connect(engine, table_name)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

# use with wfo_run, wfo_period_summary, or wfo_strategy
def connect(engine, table_name = 'wfo_run'):
    df = pd.read_sql(f'SELECT * FROM {table_name}', engine)
    return df
```

To be sure that this repo works properly, it is best to be on python 3.11.9 as vectorbt and deap are constantly being changed with various updates. Other python versions could work, but don't take my word for it. Best to use a virtual environment. 

Once you have your postgreSQL credentials initialized, run `python main.py` and the database should be populated.

It would also be best to run this in a virtual environment, but up to you. 

---

Step 3: Backtesting

Now, head over to pipeline.py to set your backtesting parameters: 

```python
start_date = '1990-01-01'   # when the first in-sample period should start
end_date = '2024-12-31'     # when the last out-of-sample period should end
random.seed(2025)           # random seed 
run_count = 5               # how many seeds to backtest on
in_sample_months = 60       # length of the in-sample period per window 
out_sample_months = 6       # length of the out-of-sample period per window
max_time_minutes = 10       # hard cap on how long processing for each window will take
stall_generations = 10      # number of generations without progress to end the training period
max_generations = 1000      # maximum number of generations for each window
pop_size = 1000             # population size per generation
n_ensemble = 50             # size of the ensemble strategy to test
leverage = 3                # leverage to use, integer from 1 to 4

custom_config = FitnessConfig(          #backtesting fitness function parameters 
    selected_metrics=['sortino', 'drawdown', 'annual_return', 'var'],
    enable_bottom_percentile_filter=True,
    bottom_percentile=10.0
)
```

Modify these values as you wish and then run the program. 