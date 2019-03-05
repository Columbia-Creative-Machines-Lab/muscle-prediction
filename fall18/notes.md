## Joint plots
We smooth force using a simple rolling average over a 1500ms time window:
$$
f_{ra} = \frac{f_{t_i-1500ms} + \dots + f_{t_i}}{n}
$$
where $n=150$ is the number of samples collected from time $t_i-1500ms$ to time $t_i$. The joint probability of smoothed force observations against temperature are pictured below. With Pearson's $\rho=0.95$ and $\rho=0.81$ we conclude that smoothed force and temperature are strongly correlated ($p<<1$). Thus we attempt to model force in response to temperature, timestamp, PWM, and previous force factors.

### LSTM
Time-series/sequential problem. Framing this as a supervised learning problem:

At time $t_i$, 

__given__:

1. sequence of temperature values $ T $ at time $(t_{i-k}) \dots (t_{i})$, $k>0$,
2. sequence of force values (N) $f$ at time $(t_{i-k}) \dots (t_{i-1})$, $k>0$, 
3. PWM input at time $t_i$,
4. timestamp $t_i$

__predict__: $f$ at $t_i$

We use mean squared error (MSE) as a loss metric:
$$
MSE = \frac{1}{2n}\sum_x \left\lVert y(x) - \hat{y}(x) \right\rVert ^ 2
$$
for a force estimate $\hat{y}(x)$ given parameters listed above at time $t_i$. The MSE loss function is commmonly used in deep learning, as it punishes large errors, preserves the units of the data, and can be computed easily.

Our model generates a rolling time-series forecast of force with high accuracy.
