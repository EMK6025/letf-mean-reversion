import pandas as pd
from sqlalchemy import text, MetaData, Table, Date, Float, Column
import numpy as np

def update_sql_table(df: pd.DataFrame, engine, input_name: str, table_name: str = 'test_data'):
    '''
    Updates the SQL table `table_name` by merging in values from `df`.
    Rows in `df` overwrite existing rows in the database, and any new dates
    are appended. The Date column is enforced as the primary key.
    '''
    df_db = pd.read_sql(f'SELECT * FROM {table_name}', engine)
    df_db['Date'] = pd.to_datetime(df_db['Date'])
    df_db.set_index('Date', inplace=True)

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df_db.update(df)

    df_db.index = pd.to_datetime(df_db.index)
    mask_new = ~df.index.isin(df_db.index)
    new_rows = df.loc[mask_new].copy()
    for col in df_db.columns:
        if col not in new_rows.columns:
            new_rows[col] = np.nan
    new_rows = new_rows[df_db.columns]

    if df_db.empty:
        combined = new_rows
    elif new_rows.empty:
        combined = df_db
    else:
        combined = pd.concat([df_db, new_rows])
    combined.reset_index(inplace=True)
    combined.drop_duplicates(subset=['Date'], keep='first', inplace=True)

    for col in combined.columns:
        if col != 'Date':
            combined[col] = pd.to_numeric(combined[col], errors='coerce').astype(float)

    combined.to_sql(table_name, engine, if_exists='replace', index=False)

    with engine.connect() as conn:
        conn.execute(text(f'''
            ALTER TABLE {table_name}
            DROP CONSTRAINT IF EXISTS {table_name}_pkey;
        '''))
        conn.execute(text(f'''
            ALTER TABLE {table_name}
            ADD PRIMARY KEY ('Date');
        '''))

    print(f'{table_name} successfully updated with {input_name} data.\n')

def clean(engine, table_name='test_data'):
    '''
    desc: cleans data by removing unnecessary entries
    in: engine connection and optional table name defaulting to test_data
    out: void, simply updates the table
    '''
    with engine.connect() as conn:
        conn.execute(text(f'''
            DELETE FROM {table_name}
            WHERE 'Date' < '1990-11-30' OR 'Date' > '2024-12-31'
        '''))
        conn.commit()
    
    print(f'entries in {table_name} before 1990-11-30 and after 2024-12-31 have been removed.\n')

def reset(engine):
    '''
    desc: resets both data tables
    in: engine connection
    out: void, simply updates the table
    '''
    with engine.connect() as conn:
        conn.execute(text('DROP TABLE IF EXISTS data'))
        conn.execute(text('DROP TABLE IF EXISTS test_data'))
        conn.commit()
        
        
    metadata = MetaData()

    data = Table( 
        'data', metadata,
        Column('Date',              Date, primary_key=True),
        Column('RF Rate',           Float),
        Column('SPXTR Change',      Float),
        Column('SPXTR Close',         Float),
        Column('2x LETF',           Float),
        Column('3x LETF',           Float),
        Column('4x LETF',           Float),
        Column('2x LETF Change',    Float),
        Column('3x LETF Change',    Float),
        Column('4x LETF Change',    Float)
    ) 

    test_data = Table( 
        'test_data', metadata,
        Column('Date',              Date, primary_key=True),
        Column('RF Rate',           Float),
        Column('SPXTR Change',      Float),
        Column('SPX Close',         Float),
        Column('2x LETF',           Float),
        Column('3x LETF',           Float),
        Column('4x LETF',           Float),
        Column('2x LETF Change',    Float),
        Column('3x LETF Change',    Float),
        Column('4x LETF Change',    Float)
    ) 

    metadata.create_all(engine)
    print(f'data and test_data have been reset to empty structures.\n')

def show(engine, table_name='test_data'):
    df = pd.read_sql(f'SELECT * from {table_name}', engine)

    with pd.option_context('display.max_columns', None):
        print('\n>>> Random 10-row sample:\n')
        print(df.sample(10))

        print('\n>>> First 5 rows:\n')
        print(df.head())

        print('\n>>> Last 5 rows:\n')
        print(df.tail())

        print('\n>>> Summary statistics:\n')
        print(df.describe())