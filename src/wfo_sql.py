from sqlalchemy import (
    Table, Column, Integer, Date, Float, JSON, 
    ARRAY, ForeignKey, CheckConstraint, 
    UniqueConstraint, MetaData, text, insert
)
import dataclasses
import pandas as pd
import numpy as np
from engine import create_engine

# wfo_run stores metadata about each run
# wfo_period_summary stores metadat about each period of a given run
# wfo_strategy stores info about each strategy with the the period and run it was trained at

metadata = MetaData()

wfo_run = Table( 
    "wfo_run",                  metadata,
    Column("run_id",            Integer,    primary_key=True,   autoincrement=True),
    Column("start_date",        Date,       nullable=False),
    Column("end_date",          Date,       nullable=False),
    Column("in_sample_months",  Integer,    nullable=False),
    Column("out_sample_months", Integer,    nullable=False),
    Column("pop_size",          Integer,      nullable=False),
    Column("n_ensemble",        Integer,      nullable=False),
    Column("leverage",          Integer,      nullable=False),
    Column("fitness_config",    JSON,       nullable=False)
) 

wfo_period_summary = Table( 
    "wfo_period_summary",       metadata,
    Column("period_id",        Integer,    primary_key=True,   autoincrement=True),
    Column("run_id",            Integer,    ForeignKey("wfo_run.run_id"), nullable=False),
    Column("period_index",      Integer,    nullable=False),
    Column("in_sample_start",   Date,       nullable=False),
    Column("in_sample_end",     Date,       nullable=False),
    Column("out_sample_end",    Date,    nullable=False),
    Column("pop_size",          Integer,      nullable=False),
    Column("n_strategies",      Integer,      nullable=False),
    UniqueConstraint("run_id", "period_index", name="unique_period_per_run")
) 

wfo_strategy = Table( 
    "wfo_strategy", metadata,
    Column("strategy_id",       Integer,        primary_key=True,   autoincrement=True),
    Column("run_id",            Integer,        nullable=False),
    Column("period_id",         Integer,        nullable=False),
    Column("window",            Integer,        nullable=False),
    Column("entry",             Integer,        nullable=False),
    Column("exit",              Integer,        nullable=False),
    Column("sell_threshold",    Float,          nullable=False),
    Column("pos_sizing",        ARRAY(Float),   nullable=False),
    Column("fitness_values",    JSON,           nullable=False),
    CheckConstraint(
        "array_length(pos_sizing, 1) = 11",
        name="ck_pos_sizing_length_11"
    )
)

def reset(engine):
    '''
    desc: resets both data tables
    in: engine connection
    out: void, simply updates the table
    '''
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS wfo_strategy"))
        conn.execute(text("DROP TABLE IF EXISTS wfo_period_summary"))
        conn.execute(text("DROP TABLE IF EXISTS wfo_run"))
        conn.commit()

    metadata.create_all(engine)
    print(f"wfo_run, wfo_period_summary, and wfo_strategy have been reset to empty structures.\n")

def insert_new_run(engine, start_date, end_date, in_sample_months, 
                   out_sample_months, pop_size, n_ensemble, 
                   leverage, fitness_config):
    input = {
            "start_date":        start_date,
            "end_date":          end_date,
            "in_sample_months":  in_sample_months,
            "out_sample_months": out_sample_months,
            "pop_size":          pop_size,
            "n_ensemble":        n_ensemble,
            "leverage":          leverage,
            "fitness_config":    dataclasses.asdict(fitness_config),
        }
    statement = insert(wfo_run).values(**input).returning(wfo_run.c.run_id)
    # needs the .c in wfo_run.c.run_id to grab that column object to return
    with engine.begin() as conn:
        result = conn.execute(statement)
        run_id = result.scalar_one() # set run_id to wfo_run.c.run_id
    return run_id

def insert_period_summary(engine, run_id, period_index, in_sample_start, 
                          in_sample_end, out_sample_end, 
                          pop_size, n_strategies):
    input = {
        "run_id": run_id,
        "period_index": period_index,
        "in_sample_start": in_sample_start, 
        "in_sample_end": in_sample_end, 
        "out_sample_end": out_sample_end, 
        "pop_size": pop_size, 
        "n_strategies": n_strategies
    }
    statement = insert(wfo_period_summary).values(**input).returning(wfo_period_summary.c.period_id)
    with engine.begin() as conn:
        result = conn.execute(statement)
        period_id = result.scalar_one() 
    return period_id

def insert_period_strategies(engine, run_id, period_id, population):
    inputs = []
    for individual in population:
        window, entry, exit, sell_threshold, *pos_sizing = individual
        inputs.append(
            {
                "run_id": run_id,
                "period_id": period_id,
                "window": window, 
                "entry": entry,
                "exit": exit, 
                "sell_threshold": sell_threshold, 
                "pos_sizing": pos_sizing,
                "fitness_values": individual.fitness.values
            }
        )

    statement = insert(wfo_strategy).values(inputs)
    with engine.begin() as conn:
        result = conn.execute(statement)
    return result.rowcount == len(inputs)

'''
CREATE TABLE wfo_run (
  run_id            SERIAL    PRIMARY KEY,
  start_date        DATE      NOT NULL,
  end_date          DATE      NOT NULL,
  in_sample_months  INT       NOT NULL,
  out_sample_months INT       NOT NULL,
  pop_size          INT       NOT NULL,
  n_ensemble        INT       NOT NULL,
  leverage          INT       NOT NULL,
  fitness_config    JSONB     NOT NULL
);

CREATE TABLE wfo_period_summary (
  period_id         SERIAL    PRIMARY KEY,
  run_id            INT       NOT NULL  REFERENCES wfo_run(run_id),
  period_index      INT       NOT NULL,
  in_sample_start   DATE      NOT NULL,
  in_sample_end     DATE      NOT NULL,
  out_sample_end    DATE      NOT NULL,
  pop_size          INT       NOT NULL,
  n_strategies      INT       NOT NULL,
  UNIQUE(run_id, period_index)
);

# one row per strategy in each ensemble
CREATE TABLE wfo_strategy (
    strategy_id       SERIAL    PRIMARY KEY,
    run_id            INT       NOT NULL,
    period_id        INT        NOT NULL,
    window            INT       NOT NULL,
    entry             INT       NOT NULL,
    exit              INT       NOT NULL,
    sell_threshold    FLOAT     NOT NULL,
    pos_sizing        FLOAT[]   NOT NULL,
    fitness_values    JSONB     NOT NULL
);
'''

if __name__ == "__main__":
    engine = create_engine()
    reset(engine)