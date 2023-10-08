import math
import time
import numpy as np
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from decimal import Decimal, ROUND_DOWN

# API credentials
api_key = 'apikey'
api_secret = 'apisecret'

# Binance client
client = Client(api_key, api_secret)

# Load the trained model
model = load_model('btc_price_prediction_model.h5')

# Initial balance and holding status
initial_balance = 20  # Initial balance in USDT
balance = Decimal(initial_balance)  # Current balance in USDT (using Decimal for precision)
holding = False  # Holding status
quantity = Decimal(0)  # Quantity of BTC held
buy_price = Decimal(0)  # Price at which BTC was bought

# Retrieve LOT_SIZE filter for BTCUSDT trading pair
symbol_info = client.get_symbol_info('BTCUSDT')
lot_size_filter = next(filter(lambda f: f['filterType'] == 'LOT_SIZE', symbol_info['filters']))

# Extract relevant values from the LOT_SIZE filter
min_qty = Decimal(lot_size_filter['minQty'])
max_qty = Decimal(lot_size_filter['maxQty'])
step_size = Decimal(lot_size_filter['stepSize'])

# Strategy parameters
price_threshold = 0.01  # Minimum price change threshold for trading
profit_ratio = 1.02  # Target profit ratio
loss_ratio = 0.98  # Stop loss ratio
# Get account information
account_info = client.get_account()
balances = account_info['balances']
usdt_balance = next((balance['free'] for balance in balances if balance['asset'] == 'USDT'), None)

if usdt_balance:
    balance = Decimal(usdt_balance)

else:
    print("USDT balance not found")


def place_sell_order(quantity):
    # Get symbol info
    symbol_info = client.get_symbol_info('BTCUSDT')
    filters = symbol_info['filters']

    # Find the LOT_SIZE filter
    lot_size_filter = next((f for f in filters if f['filterType'] == 'LOT_SIZE'), None)

    if lot_size_filter:
        # Get the minimum lot size allowed
        min_lot_size = float(lot_size_filter['minQty'])
        btc_balance = client.get_asset_balance(asset='BTC')
        btc_balance = btc_balance['free']
        # Round down the quantity to the minimum lot size
        btc_balance = float(btc_balance)
        quantity = math.floor(btc_balance / min_lot_size) * min_lot_size
        print(quantity)
        # Sell BTC
        sell_order = client.order_market_sell(
            symbol='BTCUSDT',
            quantity=0.00065
        )
        return sell_order
    else:
        # Handle case when LOT_SIZE filter is not found
        return None


def place_buy_order(quantity):
    # Place a market buy order
    usdt_balance = client.get_asset_balance(asset='USDT')
    usdt_balance = usdt_balance['free']

    # Buy BTC
    buy_order = client.order_market_buy(
        symbol='BTCUSDT',
        quoteOrderQty=usdt_balance
    )
    return buy_order


# Continuously make predictions, check the error, and make buying/selling decisions
while True:
    # Get the last 60 minutes of BTC prices from Binance
    klines = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=120)
    btc_prices = [float(kline[4]) for kline in klines]

    # Use only the 'Close' prices for prediction
    btc_close = np.array(btc_prices).reshape(-1, 1)

    # Scale the data to the range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    btc_close_scaled = scaler.fit_transform(btc_close)

    # Prepare the input data for prediction
    last_60_minutes = btc_close_scaled[-60:].reshape(1, -1, 1)

    # Make predictions
    predicted_prices = model.predict(last_60_minutes)

    # Inverse scale the predicted prices
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Get the actual price after 1 minute
    actual_price = btc_close[-1][0]

    # Calculate the error
    error = actual_price - predicted_prices[0][0]
    print("Actual Price:", actual_price)
    print("Predicted Price:", predicted_prices[0][0])
    print("Predicted Prices:", predicted_prices)
    stable = False
    if error < -0.3:
        print("Its going up", error)
    elif error > 0.3:
        print("Its going down", error)
    else:
        print('stable')

    if holding:
        btc_balance = client.get_asset_balance(asset='BTC')
        if btc_balance:
            quantity = math.floor(float(btc_balance['free']) * 100000) / 100000
            print(quantity)
        else:
            print("BTC balance not found")
        print(buy_price)
        if error >= -0.3 or actual_price - buy_price < -2:
            print("Sold Price:", actual_price)
            place_sell_order(quantity)
            holding = False

    else:
        usdt_balance = client.get_asset_balance(asset='USDT')
        if usdt_balance:
            quantity = math.floor(float(usdt_balance['free']) * 100) / 100

            print(quantity)
        else:
            print("USDT balance not found")

        if error < -0.3:
            print("Bought Price:", actual_price)
            place_buy_order(quantity)
            buy_price = actual_price
            holding = True

    time.sleep(1)
