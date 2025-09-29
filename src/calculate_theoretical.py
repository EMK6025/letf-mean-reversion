import pandas as pd
from engine import create_engine, connect_time_series
from populate_letf import theoretical_return
import numpy as np
import random
from sql_functions import update_sql_table

# annual fees for SSO, UPRO, and then hypothetical 4x LETF
FEE = [0, 0, 0.89/100/252, 0.91/100/252, 0.93/100/252]

def theoretical_change(df, X):
    return df['SPXTR Change']*X - df['RF Rate']*(X-1) - FEE[X]

seed = 2025
random.seed(seed)
np.random.seed(seed)

def main():

    # grab SSO
    sso = pd.read_csv('SSO-historical_nav.csv', usecols=['Date', 'NAV Change (%)'])
    sso = sso.rename(columns={
        'NAV Change (%)': '2x LETF Change'
    })
    sso['Date'] = pd.to_datetime(sso['Date'])
    sso['2x LETF Change'] = sso['2x LETF Change'].astype(float)
    sso.set_index('Date', inplace=True)
    sso.sort_index(inplace=True)
    
    # grab UPRO
    upro = pd.read_csv('UPRO-historical_nav.csv', usecols=['Date', 'NAV Change (%)'])
    upro = upro.rename(columns={
        'NAV Change (%)': '3x LETF Change'
    })
    upro['Date'] = pd.to_datetime(upro['Date'])
    upro['3x LETF Change'] = upro['3x LETF Change'].astype(float)
    upro.set_index('Date', inplace=True)
    upro.sort_index(inplace=True)


    engine = create_engine()
    df = connect_time_series(engine, 'test_data')
    
    df['2x LETF Change'] = theoretical_change(df, 2)
    df['3x LETF Change'] = theoretical_change(df, 3)
    df['4x LETF Change'] = theoretical_change(df, 4)
    
    naive_est_sso = df[['2x LETF Change']]
    naive_est_upro = df[['3x LETF Change']]
    
    # match up time series
    start = sso.index[0] + pd.DateOffset(days=1)
    end   = pd.to_datetime('2024-12-31')
    naive_est_sso = naive_est_sso[start:end]
    sso = sso[start:end]
    
    start = upro.index[0]+pd.DateOffset(days=1)
    naive_est_upro = naive_est_upro[start:end]
    upro = upro[start:end]
    
    
    first_layer_resid_2x = (sso['2x LETF Change'] - naive_est_sso['2x LETF Change'])
    first_layer_resid_3x = (upro['3x LETF Change'] - naive_est_upro['3x LETF Change'])
    
    bias_2x = first_layer_resid_2x.mean()
    bias_3x = first_layer_resid_3x.mean()
    first_layer_resid_2x_centered = first_layer_resid_2x - bias_2x
    first_layer_resid_3x_centered = first_layer_resid_3x - bias_3x

    # from scipy.stats import t
    # df2, loc2, scale2 = t.fit(first_layer_resid_2x_centered)
    # df3, loc3, scale3 = t.fit(first_layer_resid_3x_centered)
    
    # n_sims = 10_000
    # sim_resid_2x_centered = t.rvs(df2, loc=0, scale=scale2, size=n_sims)
    # sim_resid_3x_centered = t.rvs(df3, loc=loc3, scale=scale3, size=n_sims)

    # sim_resid_2x = sim_resid_2x_centered + bias_2x
    # sim_resid_3x = sim_resid_3x_centered + bias_3x
    
    # sim_error_2x = sso['2x LETF Change'] - naive_est_sso['2x LETF Change'] - sim_resid_2x[0:naive_est_sso.shape[0]]
    # sim_error_3x = upro['3x LETF Change'] - naive_est_upro['3x LETF Change'] - sim_resid_3x[0:naive_est_upro.shape[0]]
    
    # mean_error = sim_error_2x.mean()
    # std_error  = sim_error_2x.std()
    # print(f'student's t estimation for SSO')
    # print(f'Mean error: {mean_error:.4f},  Std dev: {std_error:.4f}')    
    
    # mean_error = sim_error_3x.mean()
    # std_error  = sim_error_3x.std()
    # print(f'student's t estimation for UPRO')
    # print(f'Mean error: {mean_error:.4f},  Std dev: {std_error:.4f}')   
    
    np_random_resid_2x = np.random.choice(first_layer_resid_2x_centered.values,
                            size=df.shape[0],
                            replace=True) + bias_2x
    np_random_resid_3x = np.random.choice(first_layer_resid_3x_centered.values,
                            size=df.shape[0],
                            replace=True) + bias_3x
    np_random_est_sso = naive_est_sso
    np_random_est_upro = naive_est_upro
    np_random_est_sso['2x LETF Change'] = naive_est_sso['2x LETF Change'] + np_random_resid_2x[0:naive_est_sso.shape[0]]
    np_random_est_upro['3x LETF Change'] = naive_est_upro['3x LETF Change'] + np_random_resid_3x[0:naive_est_upro.shape[0]]
        
    # np_random_error_2x = (sso['2x LETF Change'] - np_random_est_sso['2x LETF Change'])
    # np_random_error_3x = (upro['3x LETF Change'] - np_random_est_upro['3x LETF Change'])
    
    # mean_error = np_random_error_2x.mean()
    # std_error  = np_random_error_2x.std()
    # print(f'np random estimation for SSO')
    # print(f'Mean error: {mean_error:.4f},  Std dev: {std_error:.4f}')    
    
    # mean_error = np_random_error_3x.mean()
    # std_error  = np_random_error_3x.std()
    # print(f'np random estimation for UPRO')
    # print(f'Mean error: {mean_error:.4f},  Std dev: {std_error:.4f}')   
    df['2x LETF Change'] += np_random_resid_2x
    df['3x LETF Change'] += np_random_resid_3x
    df['4x LETF Change'] += np_random_resid_3x + bias_3x - bias_2x
    df.loc[sso.index, '2x LETF Change'] = sso['2x LETF Change']

    df.loc[upro.index, '3x LETF Change'] = upro['3x LETF Change']
    
    # made Date useable
    df.reset_index(inplace=True)  
    update_sql_table(df, engine, 'LETF Change', table_name='test_data')
    
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(figsize=(12, 6))

    # ax.plot(np_random_est_sso.index, np_random_est_sso['NAV'], label='Theoretical 2x SSO', linewidth=2)
    # ax.plot(sso.index, sso['NAV'], label='Actual SSO', linewidth=2, alpha=0.7)
    # ax.plot(np_random_est_upro.index, np_random_est_upro['NAV'], label='Theoretical 3x UPRO', linewidth=2)
    # ax.plot(upro.index, upro['NAV'], label='Actual UPRO', linewidth=2, alpha=0.7)
    
    # ax.set_title('Theoretical vs Actual Leveraged Change', fontsize=14)
    # ax.set_ylabel('NAV Value')
    # ax.set_xlabel('Date')
    # ax.grid(True, alpha=0.3)
    # ax.legend()

    # plt.tight_layout()
    # plt.show()
    