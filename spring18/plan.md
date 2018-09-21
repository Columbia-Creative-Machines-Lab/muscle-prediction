# Exploratory analysis

| Feature   | Unit |
-----------:|-----:|
| TIME			| s		 |
| T ATM			| C		 |
| T MUSCLE  | C		 |
| PWM				|	Hz	 |
| FORCE			| N		 |

- Eliminate outliers
- Correlation b/w time & `t_atm, t_muscle, force`
- Trends in: `min, max, period, cooling time, heating time`
- Trends in: `tail, belly, skew, kurtosis, area under curve, ...`
- Label cycles (`t_{min} < t_i < t_{min}`)

# Analytical solutions
- Diffusion
- Ethanol pressure
- Specific volume (volume : mass)
- `F_max` and `F_min`

# Models
- Forecasting and regression
- Fourier analysis
- ARIMA, vector autoregression, exponential smoothing

# Evaluation
- R^2 / lag

# Timeline
- __Sept.__ Exploratory analysis & feature engineering
- __Oct.__ Model evaluation, next model
- __Nov.__ Model evaluation, next model
- __Dec.__ Paper-writing
