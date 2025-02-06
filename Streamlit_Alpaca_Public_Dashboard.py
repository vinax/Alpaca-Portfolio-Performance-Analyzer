##########################################
##                                      ##
##      ALPACA PORTFOLIO DASHBOARD      ##
##         PERFORMANCE ANALYZER         ##
##           Copyright © 2025           ##
##                                      ##
##########################################

# Install necessary libraries
# pip install alpaca-trade-api pandas matplotlib streamlit numpy yfinance scipy arch scikit-learn statsmodels xlsxwriter

# Imports
import os
import alpaca_trade_api as tradeapi
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st
import datetime
import datetime as dt
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
import scipy
from scipy.stats import ttest_ind, ttest_1samp, skew, kurtosis, linregress, t, wilcoxon, norm, bootstrap
import arch
from arch.unitroot import VarianceRatio
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
import statsmodels.formula.api as smf
import io
from io import BytesIO
import json
import tempfile
import zipfile

################## MAIN ##################

st.title("Alpaca Portfolio Dashboard Performance Analyzer")
st.write("### About")
st.write("This is a basic web app/software tool designed to provide Alpaca users with a user-friendly way to obtain detailed analytic insights into their account portfolio history.")

with st.expander("ℹ️ **Read Disclaimer & Security Notice**"):
    st.write("""
    ### Disclaimer  
    This software tool is for **educational and informational purposes only**. It is provided **as-is** and is **not intended for commercial use without explicit permission**.  
    - This is **not** financial advice or a trading recommendation.  
    - The author assumes **no liability for financial losses, incorrect data, or misinterpretations** arising from the use of this tool.  
    - Users are responsible for verifying all information before making financial decisions.  

    ### Security & Privacy Notice  
    - **Your Alpaca API Key and Secret are never stored, logged, or transmitted to any third party.**  
    - All account data is **processed locally** and only retrieved for the active session.  
    - **No confidential information is collected, stored, or shared.**  
    - **Ensure your API Key and Secret remain secure** and are not exposed or auto-saved in your browser.  
    - The author is **not responsible** for unauthorized access due to poor security practices.  

    ### Third-Party Disclaimer  
    - **This software is an independent third-party tool and is not affiliated with, endorsed by, or officially connected to Alpaca Securities LLC.**  
    - **Alpaca** and any related trademarks belong to **Alpaca Securities LLC**, and their use in this project is purely for descriptive purposes.  
    - For official Alpaca services, visit: [Alpaca's Website](https://alpaca.markets).  

    ### License & Source Code  
    This software is open-source and released under the **GNU General Public License v3.0 (GPL-3.0)**.  
    You are free to **use, modify, and distribute** this software as long as any derivative work remains open-source under the same license.  
    **View the full source code here:** [GitHub Repository](https://github.com/vinax/Alpaca-Portfolio-Performance-Analyzer)  
    """)

st.write("#### If on mobile web click the top left [>] arrow to proceed.")
st.write("Please wait while your request is being processed.")

st.sidebar.header("Account Settings") # User Inputs via Streamlit Sidebar
account = st.sidebar.radio('Select Account', ['Paper', 'Live'])
key = st.sidebar.text_input('Enter Alpaca API Key', type="password")
secret = st.sidebar.text_input('Enter Alpaca Secret Key', type="password")
base_url = "https://paper-api.alpaca.markets" if account == "Paper" else "https://api.alpaca.markets" # Set base URL based on account type
os.environ["https://api.alpaca.markets"] = base_url
os.environ["APCA_API_KEY_ID"] = key
os.environ["APCA_API_SECRET_KEY"] = secret
try: # Initialize Alpaca API
    api = tradeapi.REST(
        os.getenv("APCA_API_KEY_ID"),
        os.getenv("APCA_API_SECRET_KEY"),
        base_url,
        api_version='v2'
    )
except Exception as e:
    st.error(f"Error initializing Alpaca API: {e}")

################## CHECK API KEY AND SECRET ARE VALID ##################

def is_api_key_valid():
    return bool(key.strip()) and bool(secret.strip())

def is_api_connection_valid(api):
    try:
        account_info = api.get_account()  # Try fetching account info
        return account_info is not None
    except tradeapi.rest.APIError:
        return False  # API is invalid or the wrong Paper/Live setting is used
    except Exception:
        return False  # Any other connection issue

################## DATES ##################

def get_earliest_latest_date(api): # Function to get the earliest and latest date from Alpaca's portfolio history
    try:
        history = api.get_portfolio_history(period='all', timeframe='1D').df # Fetch portfolio history for the longest available period
        if not history.empty:
            earliest_date = history.index.min().date() # Return the earliest and latest timestamp in the data
            latest_date = history.index.max().date()
            return earliest_date, latest_date
        else:
            st.warning("No portfolio history found. Using default start and end date.")
            earliest_date = dt.date(1970, 1, 1)
            latest_date = dt.date(2020, 1, 1)
            return earliest_date, latest_date
    except Exception as e:
        st.error(f"Error fetching earliest and latest date: {e}")
        earliest_date = dt.date(1970, 1, 1)
        latest_date = dt.date(2020, 1, 1)
        return earliest_date, latest_date
earliest_date, latest_date = get_earliest_latest_date(api) # Dynamically set the starting date

if "starting_date" not in st.session_state: # Initialize session state for dates
    st.session_state.starting_date = earliest_date
if "ending_date" not in st.session_state:
    st.session_state.ending_date = latest_date
if st.sidebar.button("Refresh"): # Add reset button
    st.session_state.starting_date = earliest_date
    st.session_state.ending_date = latest_date
starting_date = st.sidebar.date_input('Enter start date for trade history', value=st.session_state.starting_date, key="starting_date") # Date input fields controlled by session state
ending_date = st.sidebar.date_input('Enter end date for trade history', value=st.session_state.ending_date, key="ending_date")
if is_api_key_valid():
    earliest_date_ISO = earliest_date.strftime("%Y-%m-%dT%H:%M:%SZ") # Convert date to ISO 8601 format
    starting_date_ISO = starting_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    ending_date_ISO = ending_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    earliest_date_UNIX = int(datetime.combine(earliest_date, datetime.min.time()).timestamp()) # Convert date to UNIX format
    starting_date_UNIX = int(datetime.combine(starting_date, datetime.min.time()).timestamp())
    ending_date_UNIX = int(datetime.combine(ending_date, datetime.min.time()).timestamp())
#try: # Debug dates
###earliest_date = earliest_date
###starting_date = starting_date
###endting_date = starting_date
###earliest_date_ISO = earliest_date_ISO
###starting_date_ISO = starting_date_ISO
###ending_date_ISO = ending_date_ISO
###earliest_date_UNIX = earliest_date_UNIX
###starting_date_UNIX = starting_date_UNIX
###ending_date_UNIX = ending_date_UNIX
###st.write(earliest_date)
###st.write(starting_date)
###st.write(starting_date)
###st.write(earliest_date_ISO)
###st.write(starting_date_ISO)
###st.write(ending_date_ISO)
###st.write(earliest_date_UNIX)
###st.write(starting_date_UNIX)
###st.write(ending_date_UNIX)
#except Exception as e:
###st.error(f"Error processing date inputs: {e}")

################## GET ACTIVITIES FROM START TO END DATE ##################

def get_activities(api, after_date, until_date, page_size=100, calculate_activities=False, activity_types=None):
    try:
        aggregated_activities = { # Initialize variables
            "deposits": 0.0,
            "withdrawals": 0.0,
            "dividends": 0.0,
            "fees": 0.0,
            "jnlc": 0.0,
            "divnra": 0.0,
            "net_fill_value": 0.0,
        }
        raw_activities = [] # To store raw data
        next_page_token = None
        while True:
            response = api.get_activities( # Fetch activities with pagination
                activity_types=activity_types if activity_types else (['CSD', 'CSW', 'DIV', 'FEE', 'JNLC', 'DIVNRA', 'FILL'] if calculate_activities else None),
                after=after_date,
                until=until_date,
                page_size=page_size,
                page_token=next_page_token
            )
            if not response: # Stop if no activities are returned
                break
            raw_activities.extend([activity._raw for activity in response]) # Process raw activities
            if calculate_activities: # If calculating activities, update the aggregated totals
                for act in response:
                    activity_type = getattr(act, "activity_type", None)
                    net_amount = float(getattr(act, "net_amount", 0.0))
                    if activity_type == "CSD":  # Deposit
                        aggregated_activities["deposits"] += net_amount
                    elif activity_type == "CSW":  # Withdrawal
                        aggregated_activities["withdrawals"] += net_amount
                    elif activity_type == "DIV":  # Dividend
                        aggregated_activities["dividends"] += net_amount
                    elif activity_type == "FEE":  # Fee
                        aggregated_activities["fees"] += net_amount
                    elif activity_type == "JNLC":  # Distribution
                        aggregated_activities["jnlc"] += net_amount
                    elif activity_type == "DIVNRA":  # Non-resident alien tax on dividends
                        aggregated_activities["divnra"] += net_amount
                    elif activity_type == "FILL":  # Trade fills
                        price = float(getattr(act, "price", 0.0))
                        qty = float(getattr(act, "qty", 0.0))
                        side = getattr(act, "side", "").lower()
                        dollar_value = price * qty
                        if side == "buy":
                            aggregated_activities["net_fill_value"] += dollar_value
                        elif side == "sell":
                            aggregated_activities["net_fill_value"] -= dollar_value
            next_page_token = response[-1].id if response else None  # Update pagination token
            if not next_page_token:
                break
        if calculate_activities: # Return results based on the `calculate_activities` flag
            return aggregated_activities
        else: 
            return pd.DataFrame(raw_activities) if raw_activities else pd.DataFrame() # Ensure raw activities are returned as a DataFrame
    except Exception as e:
        print(f"Error fetching activities: {e}")
        return {} if calculate_activities else pd.DataFrame() # Return an empty structure depending on the mode

################## GET PORTFOLIO HISTORY FROM SELECTED START TO END DATE ##################

def get_portfolio_history(api, after_date, until_date):
    try:
        if after_date and until_date:
            portfolio_history = api.get_portfolio_history( # Fetch portfolio history for a specific date range
                date_start=after_date,
                date_end=until_date,
                timeframe='1D'
            )
            if portfolio_history and portfolio_history.equity:
                earliest_portfolio_value = portfolio_history.equity[0]
                latest_portfolio_value = portfolio_history.equity[-1]
                return {
                    "earliest_value": earliest_portfolio_value,
                    "latest_value": latest_portfolio_value,
                    "data": portfolio_history
                }
            else:
                st.warning("No equity data found for the specified start date.")
                return None
        else:
            history = api.get_portfolio_history(period='all', timeframe='1D').df # Fetch the full portfolio history
            history.reset_index(inplace=True)
            history['timestamp'] = pd.to_datetime(history['timestamp']).dt.date
            return {"full_history": history}
    except Exception as e:
        st.error(f"Error fetching portfolio history: {e}")
        return None

################## MERGE DATAFRAMES ##################

def process_portfolio_history(raw_data): # Process raw portfolio history into a dataframe
    try:
        history_df = pd.DataFrame({ # Extract necessary data from PortfolioHistory object
            'timestamp': pd.to_datetime(raw_data.timestamp, unit='s'),  # Convert timestamp to datetime
            'equity': raw_data.equity,
            'profit_loss': raw_data.profit_loss,
            'profit_loss_pct': raw_data.profit_loss_pct
        })
        history_df['Date'] = history_df['timestamp'].dt.date # Add a Date column for readability
        return history_df
    except Exception as e:
        st.error(f"Error processing portfolio history: {e}")
        return pd.DataFrame()

def process_activities_dataframe(activities_raw_data):
    try:
        activities_df = pd.DataFrame(activities_raw_data)  # Convert raw data to DataFrame
        if 'price' not in activities_df.columns:
            activities_df['price'] = 0.0  # Ensure price column exists
        activities_df['price'] = pd.to_numeric(activities_df['price'], errors='coerce').fillna(0)  # Convert price to float
        try: # Extract timestamp from `id` (Alpaca sometimes encodes timestamps in IDs)
            activities_df['id_timestamp'] = activities_df['id'].str.slice(0, 17)  # Extract timestamp from ID
            activities_df['timestamp'] = pd.to_datetime(
                activities_df['id_timestamp'], format='%Y%m%d%H%M%S%f'
            )
        except Exception as e:
            st.write(f"Error processing timestamp from 'id': {e}")
        activities_df['timestamp'] = activities_df['timestamp'].dt.tz_localize(None) # Remove timezone info for compatibility with portfolio history
        activities_df.sort_values(by='timestamp', ascending=True, inplace=True) # Ensure sorting by timestamp
        return activities_df
    except Exception as e:
        st.write(f"Error processing activities dataframe: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if processing fails

def merge_dataframes(portfolio_df, activities_df): # Merge two dataframes
    try:
        merged_df = pd.merge( # Merge the two DataFrames on the 'timestamp' column
            portfolio_df, activities_df, 
            on='timestamp', 
            how='outer',  # Use 'outer' to include all timestamps, adjust as needed
            suffixes=('_portfolio', '_activity')
        )
        merged_df.sort_values(by='timestamp', ascending=True, inplace=True) # Sort by timestamp for better readability
        return merged_df
    except Exception as e:
        st.error(f"Error merging dataframes: {e}")
        return None

def aggregate_merged_data(merged_df):  # Aggregate merged data
    try:
        # Step 1: Ensure 'date' column and numeric formatting
        merged_df['date'] = pd.to_datetime(merged_df['timestamp']).dt.date  # Convert 'timestamp' to date

        # Ensure required numeric columns exist before performing operations
        required_columns = ['net_amount', 'price', 'qty']
        for col in required_columns:
            if col not in merged_df.columns:
                merged_df[col] = 0  # Fill missing columns with zeros to avoid errors

        merged_df['net_amount'] = pd.to_numeric(merged_df['net_amount'], errors='coerce').fillna(0)  # Ensure 'net_amount' is numeric
        merged_df['price'] = pd.to_numeric(merged_df['price'], errors='coerce').fillna(0)  # Ensure 'price' is numeric
        merged_df['qty'] = pd.to_numeric(merged_df['qty'], errors='coerce').fillna(0)  # Ensure 'qty' is numeric

        # Step 2: Calculate 'FILL_value'
        merged_df['FILL_value'] = 0.0  # Initialize 'FILL_value' as a float column
        fill_condition = merged_df['activity_type'] == 'FILL'
        merged_df.loc[fill_condition, 'FILL_value'] = (
            merged_df.loc[fill_condition, 'price'].astype(np.float64) *
            merged_df.loc[fill_condition, 'qty'].astype(np.float64)
        )

        # Step 3: Create dynamic aggregation dictionary for activity types
        activity_types = merged_df['activity_type'].dropna().unique()  # Get unique activity types
        agg_dict = {
            f"{activity_type}_count": ('activity_type', lambda x, activity=activity_type: (x == activity).sum())
            for activity_type in activity_types
        }
        agg_dict.update({
            f"{activity_type}_net_amount": ('net_amount', lambda x, activity=activity_type: x[merged_df.loc[x.index, 'activity_type'] == activity].sum())
            for activity_type in activity_types
        })
        agg_dict['FILL_net_amount'] = ('FILL_value', 'sum')  # Sum of 'FILL_value'
        agg_dict['equity'] = ('equity', 'last')  # Take the last 'equity' value of the day

        # Step 4: Perform aggregation
        aggregated_df = merged_df.groupby('date').agg(**agg_dict).reset_index()

        # Step 5: Add missing columns with zero values if they don't exist
        expected_net_amount_columns = [
            'JNLC_net_amount', 'DIV_net_amount', 'DIVNRA_net_amount',
            'FEE_net_amount', 'CSD_net_amount', 'CSW_net_amount'
        ]
        for col in expected_net_amount_columns:
            if col not in aggregated_df.columns:
                aggregated_df[col] = 0

        # Step 6: Reorder columns for readability
        columns_order = ['date', 'equity'] + [col for col in aggregated_df.columns if col not in ['date', 'equity']]
        aggregated_df = aggregated_df[columns_order]

        # Step 7: Remove '_count' columns if unnecessary
        aggregated_df = aggregated_df.loc[:, ~aggregated_df.columns.str.endswith('_count')]

        return aggregated_df
    except Exception as e:
        st.error(f"Error aggregating data: {e}")
        return None

################## DISPLAY PORTFOLIO OVERVIEW ##################

if is_api_key_valid() and is_api_connection_valid(api) and starting_date < ending_date:

    st.header("Portfolio Overview")

    def plot_portfolio(data):
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(data['timestamp'], data['equity'], label='Equity', color='blue')
            plt.xlabel('Date')
            plt.ylabel('Equity ($)')
            plt.title('Portfolio Value Over Time')
            plt.legend()
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error plotting portfolio: {e}")
    history = get_portfolio_history(api, after_date=starting_date, until_date=ending_date) # Call the get_portfolio_history function
    if history: # Handle the response from get_portfolio_history
        if "full_history" in history:
            history["full_history"].sort_values(by="timestamp", ascending=True, inplace=True) # Sort by timestamp
            plot_portfolio(history["full_history"]) # Use the sorted portfolio history
        elif "data" in history:
            portfolio_data = history["data"].df # Use the portfolio data for the specified date range
            portfolio_data.reset_index(inplace=True)
            portfolio_data['timestamp'] = pd.to_datetime(portfolio_data['timestamp']).dt.date
            portfolio_data.sort_values(by='timestamp', ascending=True, inplace=True) # Sort by timestamp
            plot_portfolio(portfolio_data)
        else:
            st.warning("No valid data returned from the portfolio history.")
    else:
        st.error("Unable to fetch portfolio history.")

    try:
        activities = get_activities(api, after_date=starting_date, until_date=ending_date, calculate_activities=True) # Retrieve data from activities
        history = get_portfolio_history(api, after_date=starting_date, until_date=ending_date) # Retrieve data from history
        deposits = activities.get("deposits", 0.0) # Extract metrics
        withdrawals = activities.get("withdrawals", 0.0)
        dividends = activities.get("dividends", 0.0)
        fees = activities.get("fees", 0.0)
        jnlc = activities.get("jnlc", 0.0)
        divnra = activities.get("divnra", 0.0)
        net_fill_value = activities.get("net_fill_value", 0.0)
        earliest_portfolio_value = history.get("earliest_value", 0.0) # Extract portfolio values
        latest_portfolio_value = history.get("latest_value", 0.0)
        net_deposit = deposits + withdrawals # Perform calculations
        portfolio_value_change = latest_portfolio_value - earliest_portfolio_value
        net_profit = portfolio_value_change - net_deposit
        if net_deposit > 0:  # Use net deposits as the baseline when deposits exist
            net_profit_percentage = (net_profit / net_deposit) * 100
        elif earliest_portfolio_value > 0:  # Use initial portfolio value if deposits are zero
            net_profit_percentage = (net_profit / earliest_portfolio_value) * 100
        else:  # Both net_deposit and initial portfolio value are zero
            net_profit_percentage = np.nan  # Undefined percentage
        st.metric("Current Portfolio Value", f"${latest_portfolio_value:.2f}" if latest_portfolio_value else "N/A") # Display metrics
        st.metric("Initial Portfolio Value", f"${earliest_portfolio_value:.2f}")
        st.metric("Portfolio Value Change (Current - Initial)", f"${portfolio_value_change:.2f}")
        st.metric("Total Deposits", f"${deposits:.2f}")
        st.metric("Total Withdrawals", f"${withdrawals:.2f}")
        st.metric("Net Deposit (Deposits - Withdrawals)", f"${net_deposit:.2f}")
        st.metric("Total Distributions", f"${jnlc:.2f}")
        st.metric("Total Dividends", f"${dividends:.2f}")
        st.metric("Total Fees", f"${fees:.2f}")
        st.metric("Net Profit (Value Change - Net Deposits)", f"${net_profit:.2f}")
        st.metric("Net Profit (%)", f"{net_profit_percentage:.2f}%" if not np.isnan(net_profit_percentage) else "N/A")
    except Exception as e:
        st.error(f"Error calculating portfolio statistics: {e}")

################## DISPLAY POSITIONS ##################

if is_api_key_valid() and is_api_connection_valid(api) and starting_date < ending_date:

    st.header("Current Positions")
    def get_positions(api):
        try:
            positions = api.list_positions()
            portfolio = pd.DataFrame({
                'Ticker': [pos.symbol for pos in positions],
                'Current Price': [float(pos.current_price) for pos in positions],
                'Cost Basis': [float(pos.cost_basis) for pos in positions],
                'Shares': [float(pos.qty) for pos in positions],
                'Unrealized Gain ($)': [float(pos.unrealized_pl) for pos in positions],
                'Unrealized Gain (%)': [float(pos.unrealized_plpc) for pos in positions],
            })
            return portfolio
        except Exception as e:
            st.error(f"Error fetching positions: {e}")
            return pd.DataFrame()
    positions = get_positions(api)
    if not positions.empty:
        st.dataframe(positions)

################## DISPLAY TRADE HISTORY ##################

if is_api_key_valid() and is_api_connection_valid(api) and starting_date < ending_date:

    st.header("Trade History")
    try:
        trades = get_activities(api, after_date=starting_date, until_date=ending_date, activity_types=['FILL']) # Fetch only trade-related activities
        if not trades.empty:
            trades['transaction_time'] = pd.to_datetime(trades['transaction_time']) # Ensure the column is in datetime format
            trades = trades.sort_values(by='transaction_time', ascending=True)  # Sort by transaction_time
            formatted_trades = trades[
                ['transaction_time', 'symbol', 'type', 'side', 'qty', 'price']
            ].rename(columns={
                'transaction_time': 'Transaction Time',
                'symbol': 'Symbol',
                'type': 'Type',
                'side': 'Side',
                'qty': 'Quantity',
                'price': 'Price ($)',
            })
            formatted_trades['Transaction Time'] = pd.to_datetime(formatted_trades['Transaction Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            formatted_trades['Quantity'] = formatted_trades['Quantity'].astype(float).map('{:.2f}'.format)
            formatted_trades['Price ($)'] = formatted_trades['Price ($)'].astype(float).map('${:.2f}'.format)
            st.dataframe(formatted_trades)
        else:
            st.write("No trades found.")
    except Exception as e:
        st.error(f"Error fetching trades: {e}")

################## FETCH SPY DAILY PRICES ##################

if is_api_key_valid() and is_api_connection_valid(api) and starting_date < ending_date:

    def fetch_spy_prices_v2(api_key, api_secret, start_date, end_date, timeframe="1Day"):
        import requests
        base_url = "https://data.alpaca.markets/v2/stocks/SPY/bars"
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        params = {
            "start": start_date,
            "end": end_date,
            "timeframe": timeframe,
        }
        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()  # Parse the JSON response
            if "bars" in data:
                # Create a DataFrame from the data
                bars_df = pd.DataFrame(data["bars"])
                bars_df["t"] = pd.to_datetime(bars_df["t"])  # Convert timestamp to datetime
                bars_df.rename(columns={"t": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
                bars_df["Date"] = bars_df["Date"].dt.date  # Keep only the date part
                return bars_df
            else:
                st.warning("No data found for the specified date range.")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching SPY prices: {e}")
            return pd.DataFrame()

    def calculate_spy_returns(spy_df): # Calculate % Returns for SPY
        try:
            spy_df["*100 % Return"] = spy_df["Close"].pct_change()  # Calculate daily returns as a percentage
            return spy_df
        except Exception as e:
            st.error(f"Error calculating SPY returns: {e}")
            return spy_df

    def calculate_cumulative_returns(spy_df): # Calculate Cumulative % Returns for SPY
        try:
            if not spy_df.empty:
                starting_price = spy_df["Close"].iloc[0]  # Get the price at the starting date
                spy_df["Cumulative % Return"] = ((spy_df["Close"] / starting_price) - 1) * 100  # Calculate cumulative return
            return spy_df
        except Exception as e:
            st.error(f"Error calculating cumulative SPY returns: {e}")
            return spy_df

    spy_prices = fetch_spy_prices_v2(key, secret, starting_date_ISO, ending_date_ISO) # Fetch SPY prices

    if not spy_prices.empty:
        spy_prices = calculate_spy_returns(spy_prices) # Calculate % Returns
        spy_prices = calculate_cumulative_returns(spy_prices) # Calculate Cumulative % Returns
        #st.dataframe(spy_prices) # Display the table with the new '% Return' and 'Cumulative % Return' columns
        #st.subheader("SPY Cumulative % Returns Over Time") # Plot SPY Cumulative Returns Over Time
        #plt.figure(figsize=(10, 5))
        #plt.plot(spy_prices["Date"], spy_prices["Cumulative % Return"], label="SPY Cumulative % Return", color="green")
        #plt.xlabel("Date")
        #plt.ylabel("Cumulative % Return")
        #plt.title("SPY Cumulative % Returns")
        #plt.legend()
        #st.pyplot(plt)
    else:
        st.warning("No SPY data available for the selected date range.")

################## DISPLAY PORTFOLIO CUMULATIVE GROWTH CHARTS ##################

if is_api_key_valid() and is_api_connection_valid(api) and starting_date < ending_date:

    # Ensure portfolio_df and activities_df are valid before merging
    earliest_date, latest_date = get_earliest_latest_date(api) # Fetch earliest time
    portfolio_history = get_portfolio_history(api, after_date=earliest_date, until_date=latest_date)
    portfolio_df = process_portfolio_history(portfolio_history.get("data", pd.DataFrame()))

    activities_raw = get_activities(api, after_date=starting_date, until_date=ending_date)
    activities_df = process_activities_dataframe(activities_raw) if not activities_raw.empty else None

    if portfolio_df is not None and activities_df is not None:
        merged_df = merge_dataframes(portfolio_df, activities_df)

        if merged_df is not None: 
            aggregated_data = aggregate_merged_data(merged_df)

            if aggregated_data is not None:

                st.header("Portfolio Analytics")

                def filtered_table(aggregate_df, initial_date, after_date, until_date):
                    try:
                        initial_date = pd.to_datetime(initial_date) # Convert initial, after, and until dates to datetime for comparison
                        after_date = pd.to_datetime(after_date)
                        until_date = pd.to_datetime(until_date)
                        aggregate_df['date'] = pd.to_datetime(aggregate_df['date'])  # Ensure 'date' is in datetime format and filter data within the date range
                        aggregate_df = aggregate_df[(aggregate_df['date'] >= initial_date) & (aggregate_df['date'] <= until_date)]
                        filtered_df = aggregate_df.rename(columns={  # Rename columns for clarity
                            'JNLC_net_amount': 'JNLC',
                            'DIV_net_amount': 'DIV',
                            'DIVNRA_net_amount': 'DIVNRA',
                            'FEE_net_amount': 'FEE',
                            'CSD_net_amount': 'CSD',
                            'CSW_net_amount': 'CSW'
                        })
                        filtered_df = filtered_df[['date', 'equity', 'JNLC', 'DIV', 'DIVNRA', 'FEE', 'CSD', 'CSW']]  # Select only relevant columns
                        filtered_df['new_column'] = np.nan  # Initialize the new column for equity calculations
                        filtered_df.reset_index(drop=True, inplace=True)
                        filtered_df['new_column'] = filtered_df['equity']  # Start with the original equity values
                        for i in range(len(filtered_df)):
                            try:
                                equity = filtered_df.loc[i, 'equity'] if pd.notna(filtered_df.loc[i, 'equity']) else 0
                                if i == 0:  # Handle the first row explicitly
                                    if equity == 0 and len(filtered_df) > 1:
                                        next_equity = filtered_df.loc[i + 1, 'equity'] if pd.notna(filtered_df.loc[i + 1, 'equity']) else 0
                                        filtered_df.loc[i, 'new_column'] = next_equity if next_equity > 0 else 0
                                    else:
                                        filtered_df.loc[i, 'new_column'] = equity
                                elif equity > 0:
                                    filtered_df.loc[i, 'new_column'] = equity
                                else:
                                    filtered_df.loc[i, 'new_column'] = (
                                        (filtered_df.loc[i - 1, 'new_column'] if pd.notna(filtered_df.loc[i - 1, 'new_column']) else 0) +
                                        (filtered_df.loc[i - 1, 'JNLC'] if pd.notna(filtered_df.loc[i - 1, 'JNLC']) else 0) +
                                        (filtered_df.loc[i - 1, 'CSD'] if pd.notna(filtered_df.loc[i - 1, 'CSD']) else 0) +
                                        (filtered_df.loc[i - 1, 'CSW'] if pd.notna(filtered_df.loc[i - 1, 'CSW']) else 0) +
                                        (filtered_df.loc[i - 1, 'DIV'] if pd.notna(filtered_df.loc[i - 1, 'DIV']) else 0) +
                                        (filtered_df.loc[i - 1, 'FEE'] if pd.notna(filtered_df.loc[i - 1, 'FEE']) else 0) +
                                        (filtered_df.loc[i - 1, 'DIVNRA'] if pd.notna(filtered_df.loc[i - 1, 'DIVNRA']) else 0)
                                    )
                            except Exception as e:
                                st.write(f"Error at Row {i}: {type(e).__name__} - {e}")
                        filtered_df['equity'] = filtered_df['new_column']  # Replace equity column with the new column values
                        filtered_df.rename(columns={'equity': 'Equity', 'date': 'Date'}, inplace=True)  # Rename columns
                        filtered_df.drop(columns=['new_column'], inplace=True)  # Drop the temporary new_column
                        filtered_df = filtered_df.sort_values(by='Date').reset_index(drop=True) # Reset index and sort by date
                        filtered_df['Cumulative CSD'] = filtered_df['CSD'].cumsum() # Calculate cumulative columns
                        filtered_df['Cumulative CSW'] = filtered_df['CSW'].cumsum()
                        filtered_df['Net Deposit'] = filtered_df['Cumulative CSD'] + filtered_df['Cumulative CSW']
                        return_calculation_df = filtered_df[filtered_df['Date'] >= after_date].copy().reset_index(drop=True) # Filter specifically for return calculations
                        return_calculation_df['% Return'] = 0.0 # Initialize % Return and Cumulative % Return
                        for i in range(1, len(return_calculation_df)):
                            try:
                                current_equity = return_calculation_df.loc[i, 'Equity']
                                prev_equity = return_calculation_df.loc[i - 1, 'Equity']
                                prev_cumulative_csd = return_calculation_df.loc[i - 1, 'Cumulative CSD']
                                prev_csd = return_calculation_df.loc[i - 1, 'CSD']
                                prev_csw = return_calculation_df.loc[i - 1, 'CSW']
                                prev_net_deposit = return_calculation_df.loc[i - 1, 'Net Deposit']
                                second_prev_net_deposit = return_calculation_df.loc[i - 2, 'Net Deposit'] if i > 1 else 0
                                if current_equity > 0 and prev_equity > 0 and prev_cumulative_csd > 0:
                                    if prev_net_deposit == second_prev_net_deposit:
                                        return_calculation_df.loc[i, '% Return'] = (current_equity / prev_equity) - 1
                                    else:
                                        return_calculation_df.loc[i, '% Return'] = ((current_equity - prev_csd - prev_csw) / prev_equity) - 1
                            except Exception as e:
                                st.write(f"Error calculating % Return at row {i}: {e}")
                                return_calculation_df.loc[i, '% Return'] = 0
                        return_calculation_df['Cumulative % Return'] = (1 + return_calculation_df['% Return']).cumprod() - 1 # Calculate Cumulative % Return
                        filtered_df = pd.merge( # Merge the % Return and Cumulative % Return back into the full dataset
                            filtered_df,
                            return_calculation_df[['Date', '% Return', 'Cumulative % Return']],
                            on='Date',
                            how='left'
                        )
                        last_cumulative_return = filtered_df['Cumulative % Return'].iloc[-1]
                        activities = get_activities(api, after_date=starting_date, until_date=ending_date, calculate_activities=True) # Retrieve data from activities
                        history = get_portfolio_history(api, after_date=starting_date, until_date=ending_date) # Retrieve data from history
                        deposits = activities.get("deposits", 0.0) # Extract metrics
                        withdrawals = activities.get("withdrawals", 0.0)
                        earliest_portfolio_value = history.get("earliest_value", 0.0) # Extract portfolio values
                        latest_portfolio_value = history.get("latest_value", 0.0)
                        net_deposit = deposits + withdrawals # Perform calculations
                        portfolio_value_change = latest_portfolio_value - earliest_portfolio_value
                        net_profit = portfolio_value_change - net_deposit
                        if net_deposit > 0:  # Use net deposits as the baseline when deposits exist
                            net_profit_percentage = (net_profit / net_deposit) * 100
                        elif earliest_portfolio_value > 0:  # Use initial portfolio value if deposits are zero
                            net_profit_percentage = (net_profit / earliest_portfolio_value) * 100
                        else:  # Both net_deposit and initial portfolio value are zero
                            net_profit_percentage = np.nan  # Undefined percentage
                        adjusted_percentage = net_profit_percentage * 0.01
                        net_profit_percent = (filtered_df['Equity'].iloc[-1] - filtered_df['Net Deposit'].iloc[-1]) / filtered_df['Net Deposit'].iloc[-1]
                        filtered_df['Adjusted Cumulative % Return'] = (adjusted_percentage / last_cumulative_return) * filtered_df['Cumulative % Return']
                        filtered_df['Cumulative Value Return'] = 1 + filtered_df['Adjusted Cumulative % Return']
                        for i in range(1, len(filtered_df)):
                            filtered_df.loc[i, 'Adjusted % Return'] = (
                                (filtered_df.loc[i, 'Cumulative Value Return'] / filtered_df.loc[i - 1, 'Cumulative Value Return']) - 1
                            )
                        filtered_df['*100 % Return'] = filtered_df['Adjusted % Return'] # Replace original columns with adjusted values
                        filtered_df['*100 Cumulative % Return'] = filtered_df['Adjusted Cumulative % Return']
                        filtered_df.drop(columns=['Adjusted Cumulative % Return', 'Adjusted % Return', 'Cumulative % Return', '% Return'], inplace=True) # Hide unimportant columns
                        #filtered_df.drop(columns=['Cumulative Value Return'], inplace=True)
                        filtered_df['Date'] = filtered_df['Date'].dt.date  # Format to display only the date
                        after_date = after_date.date() # Ensure after_date is also a datetime.date
                        filtered_df = filtered_df[filtered_df['Date'] >= after_date].copy().reset_index(drop=True)
                        return filtered_df
                    except Exception as e:
                        st.error(f"Error creating filtered table: {e}")
                        return None

                try: # Create and process dataframes for portfolio and activities
                    earliest_date, latest_date = get_earliest_latest_date(api) # Fetch earliest time
                    portfolio_history = get_portfolio_history(api, after_date=earliest_date, until_date=ending_date) # Fetch and process portfolio history
                    if portfolio_history and "data" in portfolio_history:
                        portfolio_df = process_portfolio_history(portfolio_history["data"])
                        #st.write("Portfolio DataFrame:", portfolio_df)
                    else:
                        st.error("Portfolio history data is empty.")
                        portfolio_df = None
                    raw_activities = get_activities(api, after_date=earliest_date, until_date=ending_date) # Fetch and process activities
                    if not raw_activities.empty:
                        activities_df = process_activities_dataframe(raw_activities)
                        #st.write("Activities DataFrame:", activities_df)
                    else:
                        st.error("Activities data is empty.")
                        activities_df = None
                    if portfolio_df is not None and activities_df is not None: # Merge DataFrames
                        merged_df = merge_dataframes(portfolio_df, activities_df)
                        #st.write("Merged DataFrame:", merged_df)
                    else:
                        st.error("Either portfolio_df or activities_df is not available.")
                        merged_df = None
                    if merged_df is not None:  
                        aggregate_df = aggregate_merged_data(merged_df)

                        if aggregate_df is not None:
                            growth_calculation_table = filtered_table(aggregate_df, initial_date=earliest_date, after_date=starting_date, until_date=ending_date)
                            
                            if growth_calculation_table is None or growth_calculation_table.empty:
                                st.error("Error: Growth Calculation Table is empty or could not be created.")
                        else:
                            st.error("Error: Aggregated Data is None.")
                    else:
                        st.error("Error: Merged DataFrame is None.")
                except Exception as e:
                    st.error(f"Error in Growth Calculations: {e}")

                def plot_charts(filtered_df): # Plot Charts
                    try:
                        figures = {}
                        fig_returns = plt.figure(figsize=(10, 5))
                        plt.plot(
                            filtered_df['Date'],
                            filtered_df['*100 % Return'] * 100, # Convert to percentage
                            label='Portfolio Returns (%)',
                            color='blue'
                        )
                        plt.xlabel('Date')
                        plt.ylabel('Portfolio Returns (%)')
                        plt.title('Portfolio Returns Over Time')
                        plt.legend()
                        #st.pyplot(plt)
                        #plt.figure(figsize=(10, 5))
                        #plt.plot(
                        #    filtered_df['Date'],
                        #    filtered_df['*100 Cumulative % Return'] * 100, # Convert to percentage
                        #    label='Portfolio Cumulative Return (%)',
                        #    color='blue'
                        #)
                        #plt.xlabel('Date')
                        #plt.ylabel('Cumulative Return (%)')
                        #plt.title('Cumulative Return Over Time')
                        #plt.legend()
                        #st.pyplot(plt)
                        figures["Portfolio Returns Chart"] = fig_returns
                        return figures
                    except Exception as e:
                        st.error(f"Error plotting charts: {e}")

                def plot_combined_cumulative_chart(portfolio_df, spy_df):
                    try:
                        figures = {}

                        # SPY Returns Chart
                        fig_spy = plt.figure(figsize=(10, 5))
                        plt.plot(
                            spy_df['Date'],
                            spy_df['*100 % Return'] * 100,  # Convert to percentage
                            label='SPY Returns (%)',
                            color='green'
                        )
                        plt.xlabel('Date')
                        plt.ylabel('SPY Returns (%)')
                        plt.title('SPY Returns Over Time')
                        plt.legend()
                        plt.grid(False)
                        #st.pyplot(fig_spy)
                        figures["SPY Returns Chart"] = fig_spy

                        # Cumulative Growth Chart
                        fig_cumulative = plt.figure(figsize=(10, 5))
                        plt.plot(
                            portfolio_df["Date"],
                            portfolio_df["*100 Cumulative % Return"] * 100,  # Convert to percentage
                            label="Portfolio Cumulative Return (%)",
                            color="blue",
                        )
                        plt.plot(
                            spy_df["Date"],
                            spy_df["Cumulative % Return"],  # Convert to percentage
                            label="SPY Cumulative Return (%)",
                            color="green",
                        )
                        plt.xlabel("Date")
                        plt.ylabel("Cumulative Return (%)")
                        plt.title("Portfolio vs. SPY Cumulative % Growth")
                        plt.legend()
                        plt.grid(False)
                        #st.pyplot(fig_cumulative)
                        figures["Cumulative Growth Chart"] = fig_cumulative
                        return figures
                    except Exception as e:
                        st.error(f"Error plotting combined cumulative chart: {e}")
                        return {}

                try:  # Display Charts
                    if growth_calculation_table is not None and not growth_calculation_table.empty:
                        portfolio_returns_chart = plot_charts(growth_calculation_table)
                        combined_charts = plot_combined_cumulative_chart(growth_calculation_table, spy_prices)
                        if portfolio_returns_chart and "Portfolio Returns Chart" in portfolio_returns_chart:
                            st.subheader("Portfolio Returns Chart")
                            st.pyplot(portfolio_returns_chart["Portfolio Returns Chart"])
                        if combined_charts and "SPY Returns Chart" in combined_charts:
                            st.subheader("SPY Returns Chart")
                            st.pyplot(combined_charts["SPY Returns Chart"])
                        if combined_charts and "Cumulative Growth Chart" in combined_charts:
                            st.subheader("Cumulative Growth Chart")
                            st.pyplot(combined_charts["Cumulative Growth Chart"])
                    else:
                        st.error("Portfolio or SPY data is unavailable for analysis.")
                except Exception as e:
                    st.error(f"An error occurred while processing and plotting charts: {e}")

################## DISPLAY PERFORMANCE METRICS ##################

if is_api_key_valid() and is_api_connection_valid(api) and starting_date < ending_date:

    # Ensure portfolio_df and activities_df are valid before merging
    earliest_date, latest_date = get_earliest_latest_date(api) # Fetch earliest time
    portfolio_history = get_portfolio_history(api, after_date=earliest_date, until_date=latest_date)
    portfolio_df = process_portfolio_history(portfolio_history.get("data", pd.DataFrame()))

    activities_raw = get_activities(api, after_date=starting_date, until_date=ending_date)
    activities_df = process_activities_dataframe(activities_raw) if not activities_raw.empty else None

    if portfolio_df is not None and activities_df is not None:
        merged_df = merge_dataframes(portfolio_df, activities_df)

        if merged_df is not None: 
            aggregated_data = aggregate_merged_data(merged_df)

            if aggregated_data is not None:

                def calculate_metrics(filtered_df, spy_df=None):
                    try:
                        returns = filtered_df['*100 % Return'].dropna()
                        spy_returns = spy_df['*100 % Return'].dropna()
                        cumulative_values = filtered_df['Cumulative Value Return']
                        years = len(filtered_df) / 252
                        trades = len(returns)

                        rolling_max = cumulative_values.cummax()
                        drawdown = (cumulative_values - rolling_max) / rolling_max
                        max_drawdown = drawdown.min()
                        recovery_time = (drawdown.loc[drawdown.idxmin():] == 0).idxmax() - drawdown.idxmin()

                        cagr = ((1 + returns).prod() ** (1 / years)) - 1
                        volatility = returns.std() * np.sqrt(252)
                        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                        spy_sharpe_ratio = spy_returns.mean() / spy_returns.std() * np.sqrt(252)
                        psr = 1 - norm.cdf(1 / (1 + sharpe_ratio))
                        bsr = (sharpe_ratio * 1**2 + 0) / (1 + 1**2)
                        tracking_error = (returns - spy_returns).std() * np.sqrt(252)  # Volatility of return difference
                        info_ratio = (returns.mean() - spy_returns.mean()) / tracking_error if tracking_error != 0 else np.nan

                        returns_array = np.array(returns.dropna())  # Drop NaNs to avoid errors
                        N = len(returns_array)
                        if N < 100:
                            hurst = np.nan  # Not enough data points
                        else:
                            chunk_sizes = [2 ** i for i in range(3, int(np.log2(N)))] # Define chunk sizes for R/S calculation
                            RS = [] # Compute Rescaled Range (R/S) for each chunk size
                            for chunk in chunk_sizes:
                                num_chunks = N // chunk
                                rescaled_ranges = []
                                for i in range(num_chunks):
                                    subset = returns_array[i * chunk:(i + 1) * chunk]
                                    mean_adj_series = subset - np.mean(subset)
                                    cumulative_deviation = np.cumsum(mean_adj_series)
                                    R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                                    S = np.std(subset, ddof=1)  # Sample standard deviation
                                    rescaled_ranges.append(R / S if S > 0 else 0)
                                RS.append(np.mean(rescaled_ranges))
                            hurst = float(np.polyfit(np.log(chunk_sizes), np.log(RS), 1)[0]) # Fit the log-log relationship to estimate the Hurst exponent

                        downside_deviation = np.sqrt(np.mean(np.minimum(returns, 0) ** 2))
                        sortino_ratio = (returns.mean() * 252) / (downside_deviation * np.sqrt(252))
                        win_rate = (returns > 0).mean()
                        profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum())

                        p_win = (returns > 0).mean()
                        p_loss = 1 - p_win
                        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
                        avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0
                        expectancy = (p_win * avg_win) - (p_loss * avg_loss)

                        if spy_df is not None and not spy_df.empty:
                            merged_df = pd.merge(filtered_df[["Date", "*100 % Return"]], spy_df[["Date", "*100 % Return"]], on="Date", suffixes=("_Portfolio", "_SPY"))
                            merged_df = merged_df.dropna()
                            st.subheader("Portfolio vs. SPY Returns:")
                            st.write(merged_df)
                            beta, alpha, r_value, p_value, std_err = linregress(merged_df["*100 % Return_SPY"], merged_df["*100 % Return_Portfolio"])
                            r_squared = r_value ** 2
                            fama_alpha, fama_pval = smf.ols("Q('*100 % Return_Portfolio') ~ Q('*100 % Return_SPY')", data=merged_df).fit().params["Intercept"], smf.ols("Q('*100 % Return_Portfolio') ~ Q('*100 % Return_SPY')", data=merged_df).fit().pvalues["Intercept"]
                            t_stat, t_pval = ttest_1samp(merged_df["*100 % Return_Portfolio"], 0)
                            if len(merged_df) >= 10:
                                wilcoxon_stat, wilcoxon_pval = wilcoxon(merged_df["*100 % Return_Portfolio"] - merged_df["*100 % Return_SPY"])
                                paired_t_stat, paired_t_pval = stats.ttest_rel(merged_df["*100 % Return_Portfolio"], merged_df["*100 % Return_SPY"])
                            else:
                                wilcoxon_stat, wilcoxon_pval, paired_t_stat, paired_t_pval = (None,) * 4
                            benchmark_sharpe = spy_sharpe_ratio # Use SPY's Sharpe Ratio as the benchmark
                            sharpe_std_error = returns.std() / np.sqrt(len(returns)) # Compute the standard error of the Sharpe Ratio
                            z_stat_sharpe = sharpe_ratio / sharpe_std_error
                            z_pval_sharpe = 2 * (1 - norm.cdf(abs(z_stat_sharpe)))  # Two-tailed test
                            z_stat_sharpe_vs_spy = (sharpe_ratio - spy_sharpe_ratio) / sharpe_std_error
                            z_pval_sharpe_vs_spy = 1 - norm.cdf(z_stat_sharpe_vs_spy)  # One-tailed test (is Sharpe > SPY?)
                            z_stat_psr = sharpe_ratio / ((returns.std() * np.sqrt(252)) / np.sqrt(len(returns)))
                            z_pval_psr = 2 * (1 - norm.cdf(abs(z_stat_psr)))
                            z_stat_bpsr = (sharpe_ratio - benchmark_sharpe) / sharpe_std_error # Compute the Z-score for PSR against the benchmark
                            z_pval_bpsr = 1 - norm.cdf(z_stat_bpsr)  # One-tailed test
                            bpsr = 1 - z_pval_bpsr # Result: Probability that the Sharpe Ratio is above the SPY Sharpe Ratio
                            np.random.seed(42)
                            if len(merged_df) >= 10 and len(spy_returns_df) >= 10:  # Ensure data is valid before bootstrapping
                                # Bootstrapping Alpha (Portfolio vs. SPY Excess Return)
                                boot_samples_portfolio = np.random.choice(merged_df["*100 % Return_Portfolio"], size=(1000, len(merged_df)), replace=True) # Manually generate bootstrapped samples for Portfolio and SPY returns
                                boot_samples_spy = np.random.choice(merged_df["*100 % Return_SPY"], size=(1000, len(merged_df)), replace=True)
                                boot_alpha_values = boot_samples_portfolio.mean(axis=1) - boot_samples_spy.mean(axis=1) # Compute bootstrapped Alpha (Portfolio Returns - SPY Returns)
                                boot_ci_lower_alpha, boot_ci_upper_alpha = float(np.percentile(boot_alpha_values, [2.5, 97.5])) # Compute confidence interval for Alpha
                                boot_mean_alpha = np.mean(boot_alpha_values)
                                boot_pval_alpha = 2 * min(np.mean(boot_alpha_values >= 0), np.mean(boot_alpha_values <= 0)) # Compute p-value (two-tailed test for Alpha > 0)
                                # Bootstrapping PSR (Probability Sharpe > SPY)
                                boot_sharpe_ratios = boot_samples_portfolio.mean(axis=1) / boot_samples_portfolio.std(axis=1) * np.sqrt(252)
                                boot_sharpe_ratios_spy = boot_samples_spy.mean(axis=1) / boot_samples_spy.std(axis=1) * np.sqrt(252)
                                boot_psr_values = 1 - np.array([ # Compute PSR for each bootstrap sample (probability Sharpe > SPY)
                                    stats.norm.cdf((boot_sharpe_ratios[i] - boot_sharpe_ratios_spy[i]) / (boot_samples_portfolio.std(axis=1)[i] / np.sqrt(len(merged_df))))
                                    for i in range(1000)
                                ])
                                boot_ci_lower_psr, boot_ci_upper_psr = float(np.percentile(boot_psr_values, [2.5, 97.5])) # Compute confidence interval for PSR
                                boot_mean_psr = np.mean(boot_psr_values)
                                boot_pval_psr = 2 * min(np.mean(boot_psr_values >= 0), np.mean(boot_psr_values <= 0)) # Compute p-value (two-tailed test for PSR > 0)
                            else:
                                boot_mean_alpha, boot_ci_lower_alpha, boot_ci_upper_alpha, boot_pval_alpha, boot_mean_psr, boot_ci_lower_psr, boot_ci_upper_psr, boot_pval_psr = (None,) * 8
                            portfolio_mean = merged_df["*100 % Return_Portfolio"].mean()
                            spy_mean = merged_df["*100 % Return_SPY"].mean()
                            portfolio_variance = merged_df["*100 % Return_Portfolio"].var()
                            spy_variance = merged_df["*100 % Return_SPY"].var()
                            correlation = np.corrcoef(merged_df["*100 % Return_Portfolio"], merged_df["*100 % Return_SPY"])[0, 1]
                        else:
                            alpha, beta, r_squared, fama_alpha, fama_pval, t_stat, t_pval, wilcoxon_stat, wilcoxon_pval, boot_mean_alpha, boot_ci_lower_alpha, boot_ci_upper_alpha, boot_pval_alpha, z_stat_psr, z_pval_psr, boot_mean_psr, boot_ci_lower_psr, boot_ci_upper_psr, boot_pval_psr, portfolio_mean, spy_mean, portfolio_variance, spy_variance, correlation = (None,) * 24

                        lw_test = LedoitWolf().fit(np.array(returns).reshape(-1, 1)).shrinkage_
                        lw_pval = norm.sf(lw_test)
                        skewness_val = skew(returns)
                        kurtosis_val = kurtosis(returns)
                        sqn = (returns.mean() / returns.std()) * np.sqrt(trades)

                        # Define ideal metric thresholds
                        ideal_thresholds = {
                            "CAGR": "> 0.1",  # Compound Annual Growth Rate, annualized return
                            "Volatility": "< 0.15",  # Annualized standard deviation of returns
                            "Sharpe Ratio": "> 1.0",  # Risk-adjusted return relative to volatility
                            "SPY Sharpe Ratio": "Reference value", # Benchmark Sharpe Ratio for comparison
                            "Probabilistic Sharpe Ratio": "> 0.95",  # Probability that Sharpe Ratio is positive
                            "Benchmark-Adjusted Probabilistic Sharpe Ratio": "> 0.95",  # Probability that Sharpe Ratio is greater than SPY's Sharpe
                            "Bayesian Sharpe Ratio": "> 1.0",  # Adjusted Sharpe Ratio using Bayesian methods
                            "Information Ratio": "> 0.5", # Identifies how much the portfolio has outperformed a benchmark
                            "Hurst Exponent": "> 0.5 and < 0.8",  # Measures persistence of time series trends
                            "Sortino Ratio": "> 1.0",  # Adjusted Sharpe Ratio focusing on downside risk
                            "Calmar Ratio": "> 0.5",  # Risk-adjusted return based on max drawdown
                            "Ulcer Index": "< 5",  # Measures drawdown severity and recovery time
                            "Max Drawdown": "> -0.2",  # Largest peak-to-trough decline in portfolio value
                            "Average Drawdown": "> -0.1",  # Average percentage decline from peak values
                            "Recovery Time": "< 180",  # Number of days required to recover from max drawdown
                            "Win Rate": "> 0.5",  # Percentage of profitable trades
                            "Profit Factor": "> 1.5",  # Ratio of gross profits to gross losses
                            "Expectancy": "> 0",  # Expected return per trade
                            "Alpha": "> 0",  # Excess return over the benchmark (SPY)
                            "Beta": "< 1",  # Sensitivity of the portfolio to market movements
                            "R-squared": "> 0.5",  # Measure of variance explained by the benchmark
                            "T-test on Alpha": "> 2.0",  # Statistical significance of Alpha
                            "P-value of T-test on Alpha": "< 0.05",  # Confidence in Alpha's significance
                            "Fama-French Alpha": "> 0",  # Alpha derived from Fama-French three-factor model
                            "P-value of Fama-French Alpha": "< 0.05",  # Significance test of Fama-French Alpha
                            "Wilcoxon Signed-Rank on Alpha": "> 0",  # Rank-based statistical test for Alpha
                            "P-value of Wilcoxon Signed-Rank on Alpha": "< 0.05",  # Significance test for Wilcoxon Alpha
                            "Bootstrap on Alpha": "> 0",  # Mean bootstrapped Alpha should be positive
                            "Bootstrap CI on Alpha": "Should not include 0",  # Confidence Interval should not include zero
                            "P-value of Bootstrap on Alpha": "< 0.05",  # Statistical significance of Bootstrapped Alpha
                            "Ledoit-Wolf Test on Sharpe Ratios": "Close to 0",  # Measures Sharpe Ratio robustness
                            "P-value of Ledoit-Wolf Test on Sharpe Ratios": "< 0.05",  # Confidence in Ledoit-Wolf test
                            "Paired T-test on Sharpe Ratios": "> 2.0",  # Tests if the portfolio’s mean return is significantly different from SPY's returns
                            "P-value of Paired T-test on Sharpe Ratios": "< 0.05",  # Statistical significance of portfolio Sharpe ratio from SPY
                            "Z-test on Probabilistic Sharpe Ratio (PSR)": "> 1.96",  # Z-score significance of PSR
                            "P-value of Z-test on Probabilistic Sharpe Ratio (PSR)": "< 0.05",  # Significance of Z-test for PSR
                            "Bootstrap on PSR": "> 1.0",  # Bootstrapped PSR should be above 1 for robustness
                            "Bootstrap CI on PSR": "Should not include 0",  # Confidence Interval should not include zero
                            "P-value of Bootstrap on PSR": "< 0.05",  # Statistical significance of Bootstrapped PSR
                            "Z-test on Benchmark-Adjusted Probabilistic Sharpe Ratio (BPSR)": "> 1.96",  # Z-score significance of BPSR
                            "P-value of Z-test on Benchmark-Adjusted Probabilistic Sharpe Ratio (BPSR)": "< 0.05",  # Significance of Z-test for BPSR
                            "Correlation": "> -0.3 and < 0.3",  # Portfolio correlation with benchmark (low absolute values preferred)
                            "Portfolio Mean": "> SPY Mean",  # Portfolio’s average daily return should exceed SPY
                            "SPY Mean": "Reference value",  # SPY benchmark average daily return
                            "Portfolio Variance": "< 0.01",  # Low return variance preferred for stability
                            "SPY Variance": "Reference value",  # Variance of the benchmark (SPY)
                            "Skewness": "> 0",  # Positive skewness indicates right-tailed distribution (desirable)
                            "Kurtosis": "> 3",  # Higher kurtosis indicates heavy tails (extreme events)
                            "SQN": "> 2.5",  # System Quality Number for assessing trading strategy robustness
                        }

                        return {
                            "metrics": {
                                "CAGR": cagr if cagr is not None else None,  # Return numeric value or None
                                "Volatility": volatility if volatility is not None else None,
                                "Sharpe Ratio": sharpe_ratio if sharpe_ratio is not None else None,
                                "SPY Sharpe Ratio": spy_sharpe_ratio if spy_sharpe_ratio is not None else None,
                                "Probabilistic Sharpe Ratio": psr if psr is not None else None,
                                "Benchmark-Adjusted Probabilistic Sharpe Ratio": bpsr if bpsr is not None else None,
                                "Bayesian Sharpe Ratio": bsr if bsr is not None else None,
                                "Information Ratio": info_ratio if info_ratio is not None else None,
                                "Hurst Exponent": hurst if hurst is not None else None,
                                "Sortino Ratio": sortino_ratio if sortino_ratio is not None else None,
                                "Calmar Ratio": (cagr / abs(max_drawdown)) if max_drawdown < 0 else None,
                                "Ulcer Index": np.sqrt(np.mean(drawdown ** 2)) if drawdown is not None else None,
                                "Max Drawdown": max_drawdown if max_drawdown is not None else None,
                                "Average Drawdown": drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else None,
                                "Recovery Time": recovery_time if recovery_time is not None else None,
                                "Win Rate": win_rate if win_rate is not None else None,
                                "Profit Factor": profit_factor if profit_factor is not None else None,
                                "Expectancy": expectancy if expectancy is not None else None,
                                "Alpha": alpha if alpha is not None else None,
                                "Beta": beta if beta is not None else None,
                                "R-squared": r_squared if r_squared is not None else None,
                                "T-test on Alpha": t_stat if t_stat is not None else None,
                                "P-value of T-test on Alpha": t_pval if t_pval is not None else None,
                                "Wilcoxon Signed-Rank on Alpha": wilcoxon_stat if wilcoxon_stat is not None else None,
                                "P-value of Wilcoxon Signed-Rank on Alpha": wilcoxon_pval if wilcoxon_pval is not None else None,                        
                                "Fama-French Alpha": fama_alpha if fama_alpha is not None else None,
                                "P-value of Fama-French Alpha": fama_pval if fama_pval is not None else None,
                                "Bootstrap on Alpha": boot_mean_alpha if boot_mean_alpha is not None else None,
                                "Bootstrap CI on Alpha": (boot_ci_lower_alpha, boot_ci_upper_alpha) if (boot_ci_lower_alpha, boot_ci_upper_alpha) is not None else None,
                                "P-value of Bootstrap on Alpha": boot_pval_alpha if boot_pval_alpha is not None else None,
                                "Ledoit-Wolf Test on Sharpe Ratios": lw_test if lw_test is not None else None,
                                "P-value of Ledoit-Wolf Test on Sharpe Ratios": lw_pval if lw_pval is not None else None,
                                "Paired T-test on Sharpe Ratios": paired_t_stat if t_stat is not None else None,
                                "P-value of Paired T-test on Sharpe Ratios": paired_t_pval if paired_t_pval is not None else None,
                                "Z-test on Probabilistic Sharpe Ratio (PSR)": z_stat_psr if z_stat_psr is not None else None,
                                "P-value of Z-test on Probabilistic Sharpe Ratio (PSR)": z_pval_psr if z_pval_psr is not None else None,
                                "Bootstrap on PSR": boot_mean_psr if boot_mean_psr is not None else None,
                                "Bootstrap CI on PSR": (boot_ci_lower_psr, boot_ci_upper_psr) if (boot_ci_lower_psr, boot_ci_upper_psr) is not None else None,
                                "P-value of Bootstrap on PSR": boot_pval_psr if boot_pval_psr is not None else None,
                                "Z-test on Benchmark-Adjusted Probabilistic Sharpe Ratio (BPSR)": z_stat_bpsr if z_stat_bpsr is not None else None,
                                "P-value of Benchmark-Adjusted Z-test on Probabilistic Sharpe Ratio (BPSR)": z_pval_bpsr if z_pval_psr is not None else None,
                                "Portfolio Mean": portfolio_mean if portfolio_mean is not None else None,
                                "SPY Mean": spy_mean if spy_mean is not None else None,
                                "Portfolio Variance": portfolio_variance if portfolio_variance is not None else None,
                                "SPY Variance": spy_variance if spy_variance is not None else None,
                                "Correlation": correlation if correlation is not None else None,
                                "Skewness": skewness_val if skewness_val is not None else None,
                                "Kurtosis": kurtosis_val if kurtosis_val is not None else None,
                                "SQN": sqn if sqn is not None else None,
                            },
                            "ideal_thresholds": ideal_thresholds,
                        }

                    except Exception as e:
                        st.error(f"Error calculating metrics: {e}")
                        return {}

                def plot_drawdown_chart(returns):
                    try:
                        cumulative_val_return = (1 + returns).cumprod() # Calculate cumulative returns and drawdown
                        rolling_max = cumulative_val_return.cummax()
                        drawdown = ((cumulative_val_return - rolling_max) / rolling_max) * 100   
                        fig, ax = plt.subplots(figsize=(10, 5)) # Create the figure and axes
                        ax.plot( # Plot cumulative returns
                            returns.index,
                            (cumulative_val_return - 1) * 100,  # Convert cumulative return to percentage
                            label="Cumulative Returns (%)",
                            color="blue",
                        )
                        ax.fill_between( # Highlight drawdown as a filled area
                            returns.index,
                            0,
                            drawdown,
                            color="red",
                            alpha=0.3,
                            label="Drawdown (%)"
                        )
                        ax.set_title("Portfolio Cumulative Return vs. Drawdown") # Add labels, title, and legend
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Value (%)")
                        ax.legend()
                        #st.pyplot(fig) # Display the chart in Streamlit
                        return fig # Return the figure object for embedding
                    except Exception as e:
                        st.error(f"Error plotting drawdown chart: {e}")
                        return None

                try:
                    plot_drawdown_chart(growth_calculation_table['*100 % Return'].dropna()) # Plot drawdown chart
                    drawdown_chart = plot_drawdown_chart(growth_calculation_table['*100 % Return'].dropna())
                    if drawdown_chart:
                        st.subheader("Drawdown Chart")
                        st.pyplot(drawdown_chart)
                    st.subheader("Growth Calculation Table:")
                    st.write(growth_calculation_table)
                    if growth_calculation_table is not None:
                        results = calculate_metrics(growth_calculation_table, spy_prices)
                        metrics = results["metrics"]
                        thresholds = results["ideal_thresholds"]

                        st.header("Performance Metrics")

                        # Add a mini-header for the metrics table
                        st.markdown(
                            """
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; border-bottom: 2px solid #000; font-weight: bold;">
                                <div style="text-align: left; flex: 1;">METRIC</div>
                                <div style="text-align: left; flex: 1;">STATUS</div>
                                <div style="text-align: left; flex: 1;">RESULT</div>
                                <div style="text-align: left; flex: 1;">THRESHOLD</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        for metric, value in metrics.items():
                            ideal = thresholds.get(metric, "N/A")  # Fetch ideal threshold or default to "N/A"

                            # Determine if the metric meets the threshold
                            meets_threshold = None  # None: No evaluation, True: Meets, False: Fails
                            if metric in thresholds:
                                try:
                                    # Handle specific Portfolio Mean vs SPY Mean comparison
                                    if metric == "Portfolio Mean" and "SPY Mean" in metrics:
                                        value = metrics["Portfolio Mean"]  # Get Portfolio Mean
                                        spy_mean = metrics["SPY Mean"]  # Get SPY Mean
                                        meets_threshold = value > spy_mean
                                    # Evaluate numeric thresholds
                                    elif ">" in ideal and "<" in ideal and "and" in ideal:
                                        lower_bound = float(ideal.split(">")[1].split("and")[0].strip())
                                        upper_bound = float(ideal.split("<")[1].strip())
                                        meets_threshold = lower_bound < float(value) < upper_bound
                                    elif ">" in ideal:
                                        meets_threshold = float(value) > float(ideal.split(">")[1].replace("%", "").strip())
                                    elif "<" in ideal:
                                        meets_threshold = float(value) < float(ideal.split("<")[1].replace("%", "").strip())
                                    elif "Close to" in ideal:
                                        target_str = ideal.split("to")[1].replace("(", "").replace(")", "").strip()
                                        target = float(target_str)
                                        if target == 0:
                                            # Use an absolute tolerance when the target is zero (e.g., 1e-6)
                                            meets_threshold = abs(float(value)) < 1e-6
                                        else:
                                            # Otherwise, use 10% of the target value as tolerance
                                            meets_threshold = abs(float(value) - target) < 0.1 * abs(target)
                                    elif "Should not include 0" in ideal:
                                        meets_threshold = not (float(value[0]) <= 0 <= float(value[1]))
                                        value = f"({float(value[0]):.4f}, {float(value[1]):.4f})"
                                except (ValueError, TypeError):
                                    meets_threshold = None  # Skip non-numeric thresholds

                            # Set the status and color
                            if meets_threshold is None:
                                status = "-"
                                color = "black"
                            elif meets_threshold:
                                status = "PASS"
                                color = "green"
                            else:
                                status = "FAIL"
                                color = "red"

                            # Format the value
                            formatted_value = f"{value:.4f}" if isinstance(value, float) else value

                            # Display the metric using st.markdown for color formatting
                            st.markdown(
                                f"""
                                <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px; border-bottom: 1px solid #eee;">
                                    <div style="font-weight: bold; text-align: left; flex: 1;">{metric}</div>
                                    <div style="color: {color}; font-weight: bold; text-align: left; flex: 1;">{status}</div>
                                    <div style="font-style: italic; text-align: left; flex: 1;">{formatted_value}</div>
                                    <div style="font-size: small; text-align: left; flex: 1;">{ideal}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    st.markdown("""
                    <small>  
                    <strong>Performance Metrics Disclaimer:</strong> The performance metric thresholds used in this tool were selected based on the author's personal specifications and are <strong>not universally applicable trading criteria</strong>.  
                    Users should <strong>independently evaluate</strong> these thresholds based on their own <strong>trading strategies, risk tolerance, and investment objectives</strong>.  
                    The author makes <strong>no guarantees</strong> that these metrics will provide profitable or accurate insights for all users.  
                    Users are encouraged to <strong>adjust the thresholds as needed</strong> to align with their specific requirements.  
                    </small>  
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error calculating metrics or plotting chart: {e}")

    else:
        st.write("No data to show.")

################## SAVE ALL DASHBOARD ACCOUNT DATA ##################

if is_api_key_valid() and is_api_connection_valid(api) and starting_date < ending_date:

    # Ensure portfolio_df and activities_df are valid before merging
    earliest_date, latest_date = get_earliest_latest_date(api) # Fetch earliest time
    portfolio_history = get_portfolio_history(api, after_date=earliest_date, until_date=latest_date)
    portfolio_df = process_portfolio_history(portfolio_history.get("data", pd.DataFrame()))

    activities_raw = get_activities(api, after_date=starting_date, until_date=ending_date)
    activities_df = process_activities_dataframe(activities_raw) if not activities_raw.empty else None

    if portfolio_df is not None and activities_df is not None:
        merged_df = merge_dataframes(portfolio_df, activities_df)

        if merged_df is not None: 
            aggregated_data = aggregate_merged_data(merged_df)

            if aggregated_data is not None:

                st.header("Download Options")

                def process_portfolio_history(raw_data): # Process raw portfolio history into a dataframe
                    try:
                        history_df = pd.DataFrame({ # Extract necessary data from PortfolioHistory object
                            'timestamp': pd.to_datetime(raw_data.timestamp, unit='s'),  # Convert timestamp to datetime
                            'equity': raw_data.equity,
                            'profit_loss': raw_data.profit_loss,
                            'profit_loss_pct': raw_data.profit_loss_pct
                        })
                        #st.write("Portfolio History:", history_df)
                        history_df['Date'] = history_df['timestamp'].dt.date # Add a Date column for readability
                        return history_df
                    except Exception as e:
                        st.error(f"Error processing portfolio history: {e}")
                        return pd.DataFrame()

                def process_activities_dataframe(activities_raw_data):
                    try:
                        activities_df = pd.DataFrame(activities_raw_data)  # Convert raw data to DataFrame
                        #st.write("Activities:", activities_df)
                        try: # Extract timestamp from `id` and convert it to datetime
                            activities_df['id_timestamp'] = activities_df['id'].str.slice(0, 17) # Extract the first 17 characters from `id` (timestamp part)
                            activities_df['timestamp'] = pd.to_datetime( # Convert extracted timestamp to datetime
                                activities_df['id_timestamp'], 
                                format='%Y%m%d%H%M%S%f'  # Format matches the `id` timestamp
                            )
                        except Exception as e:
                            st.write(f"Error processing timestamp from 'id': {e}")
                        try: # Remove timezone information to match portfolio history format
                            activities_df['timestamp'] = activities_df['timestamp'].dt.tz_localize(None)
                        except Exception as e:
                            st.write(f"Error removing timezone info: {e}")
                        try:
                            activities_df.sort_values(by='timestamp', ascending=True, inplace=True) # Sort by the new `timestamp` column
                        except Exception as e:
                            st.write(f"Error sorting activities dataframe: {e}")
                        return activities_df
                    except Exception as e:
                        st.write(f"Error processing activities dataframe: {e}")
                        return None

                # Create a DataFrame for Portfolio Overview statistics
                def create_portfolio_overview_df(stats):
                    try:
                        data = {
                            "Metric": [
                                "Current Portfolio Value",
                                "Initial Portfolio Value",
                                "Portfolio Value Change (Current - Initial)",
                                "Total Deposits",
                                "Total Withdrawals",
                                "Net Deposit (Deposits - Withdrawals)",
                                "Total Distributions",
                                "Total Dividends",
                                "Total Fees",
                                "Net Profit (Value Change - Net Deposits)",
                                "Net Profit (%)",
                            ],
                            "Value": [
                                stats.get("current_value", "N/A"),
                                stats.get("initial_value", "N/A"),
                                stats.get("value_change", "N/A"),
                                stats.get("total_deposits", "N/A"),
                                stats.get("total_withdrawals", "N/A"),
                                stats.get("net_deposit", "N/A"),
                                stats.get("total_distributions", "N/A"),
                                stats.get("total_dividends", "N/A"),
                                stats.get("total_fees", "N/A"),
                                stats.get("net_profit", "N/A"),
                                f"{stats.get('net_profit_percentage', 'N/A')}%",
                            ],
                        }
                        return pd.DataFrame(data)
                    except Exception as e:
                        st.error(f"Error creating Portfolio Overview DataFrame: {e}")
                        return pd.DataFrame()

                # Create a DataFrame for Performance Metrics with Ideal Thresholds
                def create_performance_metrics_df(metrics, ideal_thresholds):
                    try:
                        data = {
                            "Metric": list(metrics.keys()),
                            "Value": [metrics[key] for key in metrics.keys()],
                            "Ideal Threshold": [ideal_thresholds.get(key, "N/A") for key in metrics.keys()],
                        }
                        return pd.DataFrame(data)
                    except Exception as e:
                        st.error(f"Error creating Performance Metrics DataFrame: {e}")
                        return pd.DataFrame()

                # Check if metrics and thresholds exist, then create the DataFrame
                performance_metrics_df = (
                    create_performance_metrics_df(results["metrics"], results["ideal_thresholds"])
                    if "results" in locals() and "metrics" in results and "ideal_thresholds" in results
                    else pd.DataFrame()
                )

                # Portfolio statistics
                portfolio_stats = {
                    "current_value": latest_portfolio_value,
                    "initial_value": earliest_portfolio_value,
                    "value_change": portfolio_value_change,
                    "total_deposits": deposits,
                    "total_withdrawals": withdrawals,
                    "net_deposit": net_deposit,
                    "total_distributions": jnlc,
                    "total_dividends": dividends,
                    "total_fees": fees,
                    "net_profit": net_profit,
                    "net_profit_percentage": f"{net_profit_percentage:.2f}" if not np.isnan(net_profit_percentage) else "N/A",
                }

                portfolio_data = get_portfolio_history(api, after_date=starting_date, until_date=ending_date) # Process portfolio history
                if portfolio_data and "data" in portfolio_data:
                    processed_portfolio = process_portfolio_history(portfolio_data["data"])
                else:
                    st.error("No portfolio history data available.")
                    processed_portfolio = pd.DataFrame()
                activities_raw_data = get_activities(api, after_date=starting_date, until_date=ending_date) # Process activities data
                if not activities_raw_data.empty:
                    processed_activities = process_activities_dataframe(activities_raw_data)
                else:
                    st.error("No activities data available.")
                    processed_activities = pd.DataFrame()
                if not processed_portfolio.empty and not processed_activities.empty: # Merge the two processed DataFrames
                    merged_data = merge_dataframes(processed_portfolio, processed_activities)
                else:
                    st.error("Unable to merge data due to missing or empty dataframes.")
                aggregated_data = aggregate_merged_data(merged_data) # Assuming merged_data is the result of the merge_dataframes function
                #if aggregated_data is not None:
                    #st.write("Aggregated Account Data:")
                    #st.dataframe(aggregated_data)

                ################## DOWNLOAD FILES ##################

                def save_all_to_excel(sheets_data, file_name="portfolio_dashboard.xlsx"):
                    try:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                            for sheet_name, df in sheets_data.items():
                                df.to_excel(writer, index=False, sheet_name=sheet_name[:31])  # Sheet names must be <= 31 chars
                        output.seek(0)
                        return output
                    except Exception as e:
                        st.error(f"Error saving all data to Excel: {e}")
                        return None

                def save_charts_to_zip(charts, zip_file_name="charts.zip"):
                    try:
                        zip_buffer = BytesIO()  # Create an in-memory ZIP file
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file: # Open the ZIP file for writing
                            for chart_name, fig in charts.items():
                                if isinstance(fig, plt.Figure):  # Ensure valid Matplotlib figure
                                    img_buffer = BytesIO() # Save the figure to an in-memory BytesIO buffer
                                    fig.savefig(img_buffer, format="png", dpi=150)
                                    img_buffer.seek(0)  # Rewind the buffer to the beginning
                                    zip_file.writestr(f"{chart_name}.png", img_buffer.read()) # Write the image to the ZIP file
                                else:
                                    st.error(f"Invalid chart object for {chart_name}: {type(fig)}")
                        zip_buffer.seek(0)  # Rewind the ZIP buffer for reading
                        return zip_buffer
                    except Exception as e:
                        st.error(f"Error creating ZIP file: {e}")
                        return None

                if 'portfolio_stats' in locals() and portfolio_stats:
                    portfolio_overview_df = create_portfolio_overview_df(portfolio_stats)
                else:
                    st.error("Error: Portfolio statistics not available.")
                    portfolio_overview_df = pd.DataFrame()  # Avoids crashing

                # Gather all data into a dictionary for export
                sheets_data = {
                    "Portfolio Overview": portfolio_overview_df,
                    "Current Positions": positions,
                    "Trade History": formatted_trades if 'formatted_trades' in locals() else pd.DataFrame(),
                    "Portfolio History": processed_portfolio,
                    "Activities": processed_activities,
                    "Aggregated Account Data": aggregated_data,
                    "Growth Calculation Table": growth_calculation_table,
                    "SPY Returns": spy_prices,
                    "Performance Metrics": performance_metrics_df,
                }

                # Combine all charts into one dictionary
                charts = {
                    "Portfolio Returns Chart": portfolio_returns_chart.get("Portfolio Returns Chart"),
                    "SPY Returns Chart": combined_charts.get("SPY Returns Chart"),
                    "Cumulative Growth Chart": combined_charts.get("Cumulative Growth Chart"),
                    "Drawdown Chart": drawdown_chart,
                }

                # Button to download the compiled Excel file with charts embedded
                if st.button("Download All Data"):
                    valid_charts = {k: v for k, v in charts.items() if isinstance(v, plt.Figure)}
                    excel_file = save_all_to_excel(sheets_data)
                    if excel_file:
                        st.download_button(
                            label="Save Report As Excel",
                            data=excel_file,
                            file_name="portfolio_dashboard.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    zip_file = save_charts_to_zip(valid_charts)
                    if zip_file:
                        st.download_button(
                            label="Save Charts As Zip",
                            data=zip_file,
                            file_name="charts.zip",
                            mime="application/zip"
                        )

    else:
        st.write("No data to show.")

else:
    st.error("API or dates are invalid or the wrong Paper/Live setting is used.")
