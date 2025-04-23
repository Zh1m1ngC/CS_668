API_KEY = os.getenv('POLYGON_API_KEY', '')
BASE_URL = "https://api.polygon.io"

# Data mining step
def fetch_minute_data(ticker, from_date, to_date):
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}"
    params = {"apiKey": API_KEY, "adjusted": "true", "sort": "asc", "limit": 50000}

    r = requests.get(url, params=params)
    time.sleep(30)
    r.raise_for_status()

    data = r.json()
    if not data.get("results"):
        return pd.DataFrame()

    df = pd.DataFrame(data["results"])
    df['t'] = pd.to_datetime(df['t'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    return df.rename(columns={'t': 'datetime'})

end_date = datetime.now(pytz.timezone('America/New_York')).date()
start_date = end_date - timedelta(days=30) # Only pulls past 30 days of data since free tier of polygon.io has a 50,000 limit
from_date = start_date.strftime('%Y-%m-%d')
to_date = end_date.strftime('%Y-%m-%d')

print("Fetching equity data...")
vti_df = fetch_minute_data("VTI", from_date, to_date)
voo_df = fetch_minute_data("VOO", from_date, to_date)

print("Fetching crypto data...")
btc_df = fetch_minute_data("X:BTCUSD", from_date, to_date)

def compute_equity_opening_performance(df):
    df = df.set_index('datetime').sort_index()
    df_window = df.between_time("09:30", "11:30")

    performance = {}
    for day, group in df_window.groupby(df_window.index.date):
        if group.empty:
            continue
        open_price = group.iloc[0]['o']
        close_price = group.iloc[-1]['c']
        performance[day] = (close_price - open_price) / open_price

    return pd.Series(performance, name="performance")

vti_perf = compute_equity_opening_performance(vti_df)
voo_perf = compute_equity_opening_performance(voo_df)

def compute_crypto_overnight_performance(df):
    df = df.set_index('datetime').sort_index()
    performance = {}

    current = start_date
    while current <= end_date:
        start_window = datetime.combine(current - timedelta(days=1), datetime.min.time()).replace(hour=20, tzinfo=pytz.timezone('America/New_York'))
        end_window = datetime.combine(current, datetime.min.time()).replace(hour=9, tzinfo=pytz.timezone('America/New_York'))

        group = df[start_window:end_window]
        if not group.empty:
            open_price = group.iloc[0]['o']
            close_price = group.iloc[-1]['c']
            performance[current] = (close_price - open_price) / open_price

        current += timedelta(days=1)

    return pd.Series(performance, name="performance")

btc_perf = compute_crypto_overnight_performance(btc_df)

# Separate analysis for VTI and VOO
vti_combined = pd.concat([vti_perf, btc_perf], axis=1, keys=["VTI", "BTC"]).dropna()
voo_combined = pd.concat([voo_perf, btc_perf], axis=1, keys=["VOO", "BTC"]).dropna()

# Models
def analyze_relationship(asset_df, asset_name):
    print(f"\nAnalyzing {asset_name}...")

    # Holt-Winters (trends and seasonality)
    hw_model = ExponentialSmoothing(asset_df[asset_name], seasonal_periods=7, trend='add', seasonal='add', initialization_method="estimated")
    hw_fit = hw_model.fit()
    print(hw_fit.summary())

    # ARIMA (time-series forcasting) https://people.duke.edu/~rnau/411arim.htm#arima100
    arima_model = ARIMA(asset_df[asset_name], order=(1,0,1)) # (p,d,q), p is the number of autoregressive terms, d is the number of nonseasonal differences needed for stationarity, and q is the number of lagged forecast errors in the prediction equation.
    arima_fit = arima_model.fit()
    print(arima_fit.summary())

    # Random Forest
    X_train, X_test, y_train, y_test = train_test_split(asset_df[['BTC']], asset_df[asset_name], test_size=0.3, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    print(f"Random Forest Training R² Score for {asset_name}:", rf.score(X_train, y_train))
    print(f"Random Forest Testing R² Score for {asset_name}:", rf.score(X_test, y_test))
    print(f"Random Forest Feature Importances for {asset_name}:", rf.feature_importances_)

    # Predict next day's performance
    latest_btc_perf = pd.DataFrame(asset_df[['BTC']].iloc[-1]).T
    rf_pred = rf.predict(latest_btc_perf)

    hw_pred = hw_fit.forecast(1)
    arima_pred = arima_fit.forecast(1)

    print(f"Predicted {asset_name} change using Holt-Winters: {hw_pred[0]:.6f}")
    print(f"Predicted {asset_name} change using ARIMA: {arima_pred[0]:.6f}")
    print(f"Predicted {asset_name} change using Random Forest: {rf_pred[0]:.6f}")

    # interpret_results
    interpret_results(asset_name, hw_pred, arima_pred, rf_pred)

def interpret_results(asset_name, hw_pred, arima_pred, rf_pred):
    interpretation = f"\nInterpretation for {asset_name}:\n"
    interpretation += f"- The Holt-Winters model predicts a {hw_pred[0]:.4%} change in {asset_name}.\n"
    interpretation += f"- The ARIMA model predicts a {arima_pred[0]:.4%} change in {asset_name}.\n"
    interpretation += f"- The Random Forest model predicts a {rf_pred[0]:.4%} change in {asset_name}.\n"

    avg_prediction = (hw_pred[0] + arima_pred[0] + rf_pred[0]) / 3
    if avg_prediction > 0:
        interpretation += f"Overall, the models suggest a potential upward movement in {asset_name}.\n"
    elif avg_prediction < 0:
        interpretation += f"Overall, the models suggest a potential downward movement in {asset_name}.\n"
    else:
        interpretation += f"The models provide mixed signals, suggesting no clear trend for {asset_name}.\n"

    print(interpretation)

analyze_relationship(vti_combined, "VTI")
analyze_relationship(voo_combined, "VOO")

# Auto tune
def parameter_tuning(asset_df, asset_name):
    """
    Tests different parameter configurations for each model and compares performance against original settings.
    Metrics tracked: AIC (Holt-Winters, ARIMA) and R² (Random Forest). Lower AIC or higher R² indicates improvement.
    """
    print(f"\n{'#'*30}")
    print(f"Parameter Tuning for {asset_name}")
    print(f"{'#'*30}")

    # Holt-Winters Tuning
    print("\nTuning Holt-Winters Model:")
    # Original configuration
    hw_original = ExponentialSmoothing(
        asset_df[asset_name],
        seasonal_periods=7,
        trend='add',
        seasonal='add',
        initialization_method="estimated"
    )
    hw_original_fit = hw_original.fit()
    original_aic = hw_original_fit.aic
    print(f"Original AIC: {original_aic:.2f}")

    # Test alternative configurations
    hw_configs = [
        {'seasonal_periods': 5, 'trend': 'add', 'seasonal': 'add'},  # Shorter seasonal cycle
        {'seasonal_periods': 7, 'trend': 'mul', 'seasonal': 'mul'},  # Multiplicative components
    ]

    for config in hw_configs:
        try:
            model = ExponentialSmoothing(
                asset_df[asset_name],
                **config,
                initialization_method="estimated"
            )
            fit = model.fit()
            new_aic = fit.aic
            improvement = original_aic - new_aic  # Positive = improvement
            print(f"Config {config}: AIC={new_aic:.2f} ({'↑' if improvement <0 else '↓'} {abs(improvement):.2f})")
        except Exception as e:
            print(f"Config {config} failed: {str(e)}")

    # ARIMA Tuning
    print("\nTuning ARIMA Model:")
    # Original configuration
    arima_original = ARIMA(asset_df[asset_name], order=(1,0,1))
    arima_original_fit = arima_original.fit()
    original_aic = arima_original_fit.aic
    print(f"Original AIC: {original_aic:.2f}")

    # Test alternative orders
    arima_orders = [
        (2,0,2),  # More autoregressive/moving average terms
        (1,1,1),  # Introduce differencing
    ]

    for order in arima_orders:
        try:
            model = ARIMA(asset_df[asset_name], order=order)
            fit = model.fit()
            new_aic = fit.aic
            improvement = original_aic - new_aic
            print(f"Order {order}: AIC={new_aic:.2f} ({'↑' if improvement <0 else '↓'} {abs(improvement):.2f})")
        except Exception as e:
            print(f"Order {order} failed: {str(e)}")

    # Random Forest Tuning
    print("\nTuning Random Forest Model:")
    # Reuse original split for fair comparison
    X = asset_df[['BTC']]
    y = asset_df[asset_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Original configuration
    rf_original = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_original.fit(X_train, y_train)
    original_r2 = rf_original.score(X_test, y_test)
    print(f"Original R²: {original_r2:.4f}")

    # Test alternative configurations
    rf_configs = [
        {'n_estimators': 200},  # More trees for potential better generalization
        {'n_estimators': 100, 'max_depth': 5},  # Limit tree depth to prevent overfitting
    ]

    for config in rf_configs:
        model = RandomForestRegressor(**config, random_state=42)
        model.fit(X_train, y_train)
        new_r2 = model.score(X_test, y_test)
        improvement = new_r2 - original_r2
        print(f"Config {config}: R²={new_r2:.4f} ({'+' if improvement >=0 else ''}{improvement:.4f})")

# Execute tuning after original analysis
print("\n\n\n###### PERFORMING PARAMETER TUNING ######")
parameter_tuning(vti_combined, "VTI")
parameter_tuning(voo_combined, "VOO")

# Autotune and autorefine
def refine_and_reanalyze(asset_df, asset_name):
    """
    Automatically selects best parameters from tuning results and re-runs analysis
    Compares original vs refined model performance metrics
    """
    print(f"\n{'#'*30}")
    print(f"Refined Analysis for {asset_name}")
    print(f"{'#'*30}")

    # Store best parameters and performance metrics
    best_params = {
        'Holt-Winters': {'params': None, 'aic': np.inf},
        'ARIMA': {'params': None, 'aic': np.inf},
        'RandomForest': {'params': None, 'r2': -np.inf}
    }

    # Parameter Space Exploration
 
    # Holt-Winters Tuning
    hw_configs = [
        {'seasonal_periods': 7, 'trend': 'add', 'seasonal': 'add'},  # Original
        {'seasonal_periods': 5, 'trend': 'add', 'seasonal': 'add'},
        {'seasonal_periods': 7, 'trend': 'mul', 'seasonal': 'mul'},
    ]

    for config in hw_configs:
        try:
            model = ExponentialSmoothing(asset_df[asset_name], **config, initialization_method="estimated")
            fit = model.fit()
            if fit.aic < best_params['Holt-Winters']['aic']:
                best_params['Holt-Winters'] = {'params': config, 'aic': fit.aic}
        except:
            continue

    # ARIMA Tuning
    arima_orders = [
        (1,0,1),  # Original
        (2,0,2),
        (1,1,1),
    ]

    for order in arima_orders:
        try:
            model = ARIMA(asset_df[asset_name], order=order)
            fit = model.fit()
            if fit.aic < best_params['ARIMA']['aic']:
                best_params['ARIMA'] = {'params': order, 'aic': fit.aic}
        except:
            continue

    # Random Forest Tuning
    X_train, X_test, y_train, y_test = train_test_split(
        asset_df[['BTC']], asset_df[asset_name], test_size=0.3, random_state=42
    )

    rf_configs = [
        {'n_estimators': 100},  # Original
        {'n_estimators': 200},
        {'n_estimators': 100, 'max_depth': 5},
    ]

    for config in rf_configs:
        try:
            model = RandomForestRegressor(**config, random_state=42)
            model.fit(X_train, y_train)
            r2 = model.score(X_test, y_test)
            if r2 > best_params['RandomForest']['r2']:
                best_params['RandomForest'] = {'params': config, 'r2': r2}
        except:
            continue

    # Re-analysis with Best Params
    
    print("\nRefined Model Configurations:")
    for model, params in best_params.items():
        print(f"{model}: {params['params']}")

    # Re-run analysis with best parameters
    print("\nRefined Model Performance:")

    # Holt-Winters Refined
    hw_refined = ExponentialSmoothing(
        asset_df[asset_name],
        **best_params['Holt-Winters']['params'],
        initialization_method="estimated"
    )
    hw_refined_fit = hw_refined.fit()
    hw_refined_pred = hw_refined_fit.forecast(1)

    # ARIMA Refined
    arima_refined = ARIMA(asset_df[asset_name], order=best_params['ARIMA']['params'])
    arima_refined_fit = arima_refined.fit()
    arima_refined_pred = arima_refined_fit.forecast(1)

    # Random Forest Refined
    rf_refined = RandomForestRegressor(**best_params['RandomForest']['params'], random_state=42)
    rf_refined.fit(X_train, y_train)
    latest_btc_perf = pd.DataFrame(asset_df[['BTC']].iloc[-1]).T
    rf_refined_pred = rf_refined.predict(latest_btc_perf)

    # Performance Comparison

    # Original model metrics (from previous analysis)
    original_metrics = {
        'Holt-Winters': {'aic': ExponentialSmoothing(
            asset_df[asset_name], seasonal_periods=7, trend='add', seasonal='add'
        ).fit().aic},
        'ARIMA': {'aic': ARIMA(asset_df[asset_name], order=(1,0,1)).fit().aic},
        'RandomForest': {'r2': RandomForestRegressor(n_estimators=100).fit(X_train, y_train).score(X_test, y_test)}
    }

    print("\nPerformance Comparison (Original vs Refined):")
    print(f"{'Model':<15} | {'Metric':<10} | {'Original':<10} | {'Refined':<10} | {'Improvement':<12}")
    print("-"*65)

    # Holt-Winters comparison
    orig_aic = original_metrics['Holt-Winters']['aic']
    refined_aic = best_params['Holt-Winters']['aic']
    print(f"{'Holt-Winters':<15} | {'AIC':<10} | {orig_aic:<10.2f} | {refined_aic:<10.2f} | {(orig_aic - refined_aic):+7.2f}")

    # ARIMA comparison
    orig_aic = original_metrics['ARIMA']['aic']
    refined_aic = best_params['ARIMA']['aic']
    print(f"{'ARIMA':<15} | {'AIC':<10} | {orig_aic:<10.2f} | {refined_aic:<10.2f} | {(orig_aic - refined_aic):+7.2f}")

    # Random Forest comparison
    orig_r2 = original_metrics['RandomForest']['r2']
    refined_r2 = best_params['RandomForest']['r2']
    print(f"{'Random Forest':<15} | {'R²':<10} | {orig_r2:<10.4f} | {refined_r2:<10.4f} | {(refined_r2 - orig_r2):+7.4f}")
    # Refined predictions interpretation
    interpret_results(f"Refined {asset_name}", hw_refined_pred, arima_refined_pred, rf_refined_pred)

# Execute refined analysis after parameter tuning
print("\n\n\n###### PERFORMING REFINED ANALYSIS ######")
refine_and_reanalyze(vti_combined, "VTI")
refine_and_reanalyze(voo_combined, "VOO")