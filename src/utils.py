import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


np.random.seed(42)


def plot_history(series, title, ylabel):
    """
    Рисует график исторического ряда (ставка или курс).

    series : pd.Series
        Ряд с индексом-датой.
    title : str
        Заголовок графика.
    ylabel : str
        Подпись по оси Y.
    """
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values)
    ax.set_title(title)
    ax.set_xlabel("Дата")
    ax.set_ylabel(ylabel)
    plt.show()


# function to estimate parameters
def ols_cir(data, dt):

    # define variables
    Nsteps = len(data)
    rs = data[:Nsteps - 1]  
    rt = data[1:Nsteps]
    
    # model initialization
    model = LinearRegression()

    # feature engineering to fit the theoretical model
    y = (rt - rs) / np.sqrt(rs)
    z1 = dt / np.sqrt(rs)
    z2 = dt * np.sqrt(rs)
    X = np.column_stack((z1, z2))

    # fit the model
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    # calculate the predicted values (y_hat), residuals and the parameters
    y_hat = model.predict(X)
    residuals = y - y_hat
    beta1 = model.coef_[0]        
    beta2 = model.coef_[1]

    # get the parameter of interest for CIR
    k0 = -beta2
    theta0 = beta1/k0
    sigma0 = np.std(residuals)/np.sqrt(dt)
    
    return k0, theta0, sigma0


import numpy as np

def simulate_cir_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    dt: float,
    n_paths: int,
    seed: int | None = None
):
    """
    Симуляция CIR-модели для одной валюты (RUB или USD).

    r0     - начальная ставка
    kappa  - скорость возврата к среднему
    theta  - долгосрочное среднее
    sigma  - волатильность
    T      - горизонт моделирования (в годах)
    dt     - шаг по времени (в годах)
    n_paths - количество сценариев
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)
    rates = np.zeros((n_paths, n_steps + 1))
    rates[:, 0] = r0

    for t in range(n_steps):
        Z = np.random.normal(size=n_paths)
        r_t = rates[:, t]
        # full truncation, чтобы избежать отрицательных ставок
        sqrt_r = np.sqrt(np.maximum(r_t, 0.0))
        dr = kappa * (theta - r_t) * dt + sigma * sqrt_r * np.sqrt(dt) * Z
        rates[:, t + 1] = r_t + dr
        # по желанию можно ещё обнулить отрицательные значения:
        rates[:, t + 1] = np.maximum(rates[:, t + 1], 0.0)

    return rates


def simulate_market_paths(
    N_sim,
    N_steps,
    T_years,
    r0_rub,
    r0_usd,
    s0_fx,
    cir_rub_params,
    cir_usd_params,
    fx_vol,
    corr_matrix,
):
    """
    Симулирует динамику трех риск-факторов:
    - краткосрочная ставка RUB (CIR)
    - краткосрочная ставка USD (CIR)
    - обменный курс USD/RUB (логнормальная модель с дрейфом r_rub - r_usd).

    Parameters
    ----------
    N_sim : int
        Количество сценариев Монте-Карло.
    N_steps : int
        Количество шагов по времени.
    T_years : float
        Горизонт моделирования в годах.
    r0_rub, r0_usd : float
        Начальные значения краткосрочных ставок (в долях, не в %).
    s0_fx : float
        Начальное значение курса USD/RUB.
    cir_rub_params : tuple (kappa_r, theta_r, sigma_r)
    cir_usd_params : tuple (kappa_d, theta_d, sigma_d)
    fx_vol : float
        Годовая волатильность логнормальной модели для курса.
    corr_matrix : array-like shape (3, 3)
        Корреляционная матрица для (RUB, USD, FX).

    Returns
    -------
    rub_rates : np.ndarray shape (N_steps+1, N_sim)
    usd_rates : np.ndarray shape (N_steps+1, N_sim)
    fx_rates  : np.ndarray shape (N_steps+1, N_sim)
    discount_integrals : np.ndarray shape (N_sim,)
        Интеграл ставки RUB по времени для каждого сценария (для дисконтирования).
    """
    dt = T_years / N_steps
    kappa_r, theta_r, sigma_r = cir_rub_params
    kappa_d, theta_d, sigma_d = cir_usd_params

    rub = np.zeros((N_steps + 1, N_sim))
    usd = np.zeros((N_steps + 1, N_sim))
    fx  = np.zeros((N_steps + 1, N_sim))
    disc_int = np.zeros(N_sim)

    rub[0, :] = r0_rub
    usd[0, :] = r0_usd
    fx[0, :]  = s0_fx

    corr_matrix = np.array(corr_matrix)
    L = np.linalg.cholesky(corr_matrix)

    for t in range(N_steps):
        # генерируем независимые нормальные
        Z = np.random.normal(size=(3, N_sim))
        # делаем их скоррелированными
        dW = np.sqrt(dt) * (L @ Z)

        r_rub = rub[t, :]
        r_usd = usd[t, :]
        s     = fx[t, :]

        # CIR для RUB
        dr_rub = (kappa_r * (theta_r - r_rub) * dt +
                  sigma_r * np.sqrt(np.maximum(r_rub, 0)) * dW[0, :])

        # CIR для USD
        dr_usd = (kappa_d * (theta_d - r_usd) * dt +
                  sigma_d * np.sqrt(np.maximum(r_usd, 0)) * dW[1, :])

        # FX под риск-нейтральной мерой в RUB: дрейф = r_rub - r_usd
        dlnS = ((r_rub - r_usd - 0.5 * fx_vol**2) * dt +
                fx_vol * dW[2, :])

        rub[t+1, :] = np.maximum(r_rub + dr_rub, 0.0)
        usd[t+1, :] = np.maximum(r_usd + dr_usd, 0.0)
        fx[t+1, :]  = s * np.exp(dlnS)

        # интеграл ставки для дисконтирования
        disc_int += r_rub * dt

    return rub, usd, fx, disc_int



import datetime as dt

def price_range_accrual(
    notional,
    lower_barrier,
    upper_barrier,
    start_date,
    end_date,
    r0_rub,
    r0_usd,
    s0_fx,
    cir_rub_params,
    cir_usd_params,
    fx_vol,
    corr_matrix,
    N_sim=10000,
    steps_per_year=252,
):
    """
    Оценивает справедливую стоимость Range Accrual на основе моделирования
    курса USD/RUB и краткосрочной ставки в RUB.

    Параметры продукта:
    -------------------
    notional : float
        Номинал сделки (максимальная выплата).
    lower_barrier : float or None
        Нижняя граница диапазона (None, если нет).
    upper_barrier : float or None
        Верхняя граница диапазона (None, если нет).
    start_date, end_date : datetime.date
        Дата начала и окончания контракта.

    Остальные параметры:
    такие же, как в simulate_market_paths.
    """
    # считаем T в годах
    days = (end_date - start_date).days
    T_years = days / 365.0
    N_steps = max(1, int(steps_per_year * T_years))

    rub, usd, fx, disc_int = simulate_market_paths(
        N_sim=N_sim,
        N_steps=N_steps,
        T_years=T_years,
        r0_rub=r0_rub,
        r0_usd=r0_usd,
        s0_fx=s0_fx,
        cir_rub_params=cir_rub_params,
        cir_usd_params=cir_usd_params,
        fx_vol=fx_vol,
        corr_matrix=corr_matrix,
    )

    # убираем начальный момент, оставляем только "рабочие" дни
    S_paths = fx[1:, :]  # shape (N_steps, N_sim)

    # условие попадания в диапазон
    in_range = np.ones_like(S_paths, dtype=bool)
    if lower_barrier is not None:
        in_range &= S_paths >= lower_barrier
    if upper_barrier is not None:
        in_range &= S_paths <= upper_barrier

    # доля времени в диапазоне по каждому сценарию
    frac_in_range = in_range.mean(axis=0)

    # дисконтирование по интегралу ставки RUB
    discount_factor = np.exp(-disc_int)

    payoff = notional * frac_in_range
    pv_scenarios = payoff * discount_factor

    fair_value = pv_scenarios.mean()
    std_error = pv_scenarios.std(ddof=1) / np.sqrt(N_sim)

    return fair_value, std_error


import numpy as np

def simulate_risk_factors(n_paths=10000, T=3, dt=1/252, 
                          r0_rub=0.075, k_rub=1.0, theta_rub=0.07, sigma_rub=0.15,
                          r0_usd=0.001, k_usd=0.5, theta_usd=0.02, sigma_usd=0.05,
                          S0=74, sigma_fx=0.10,
                          corr_matrix=np.array([[1.0, 0.2, -0.3],
                                                 [0.2, 1.0, 0.2],
                                                 [-0.3, 0.2, 1.0]])):
    """
    Симуляция путей для r_RUB(t), r_USD(t) и S(t) на интервале [0, T] лет.
    Возвращает кортеж из трех массивов numpy: (paths_rub, paths_usd, paths_fx),
    каждый размером (n_paths, n_steps+1), включая начальное значение.
    """
    np.random.seed(0)  # фиксируем seed для воспроизводимости (можно убрать при реальных прогонах)
    n_steps = int(T / dt)
    
    # Резервируем массивы для результатов
    paths_rub = np.zeros((n_paths, n_steps+1))
    paths_usd = np.zeros((n_paths, n_steps+1))
    paths_fx  = np.zeros((n_paths, n_steps+1))
    
    # Устанавливаем начальные значения
    paths_rub[:, 0] = r0_rub
    paths_usd[:, 0] = r0_usd
    paths_fx[:, 0]  = S0
    
    # Предвычисляем матрицу для корреляции (Холецкого)
    L = np.linalg.cholesky(corr_matrix)
    
    # Проходим по шагам времени
    for t in range(1, n_steps+1):
        # Генерируем независимые стандартизованные случайные величины для всех путей (размер n_paths x 3)
        Z = np.random.normal(size=(n_paths, 3))
        # Вводим корреляцию: умножаем независимые N(0,1) на матрицу L (нижнетреугольную)
        # Каждый вектор Z[i] (1x3) преобразуется в коррелированный X[i] = Z[i] * L^T
        Z_corr = Z.dot(L.T)
        # Разделяем на компоненты для удобства
        Z_rub = Z_corr[:, 0]   # коррелированный шум для r_RUB
        Z_usd = Z_corr[:, 1]   # коррелированный шум для r_USD
        Z_fx  = Z_corr[:, 2]   # коррелированный шум для S (FX)
        
        # Текущее значение процессов на шаге t-1
        r_rub_prev = paths_rub[:, t-1]
        r_usd_prev = paths_usd[:, t-1]
        S_prev     = paths_fx[:, t-1]
        
        # Обновляем процентные ставки по схеме Эйлера для CIR
        # r_t = r_prev + k*(theta - r_prev)*dt + sigma*sqrt(r_prev)*sqrt(dt)*Z
        # Сразу применяем max(..., 0) внутри sqrt для безопасности
        r_rub_new = r_rub_prev + k_rub * (theta_rub - r_rub_prev) * dt \
                    + sigma_rub * np.sqrt(np.maximum(r_rub_prev, 0)) * np.sqrt(dt) * Z_rub
        r_usd_new = r_usd_prev + k_usd * (theta_usd - r_usd_prev) * dt \
                    + sigma_usd * np.sqrt(np.maximum(r_usd_prev, 0)) * np.sqrt(dt) * Z_usd
        
        # Обновляем обменный курс по логнормальной модели
        # Используем актуальные ставки для вычисления дрейфа
        drift = (r_rub_prev - r_usd_prev)  # риск-нейтральный дрейф на текущем шаге
        S_new = S_prev * np.exp((drift - 0.5 * sigma_fx**2) * dt + sigma_fx * np.sqrt(dt) * Z_fx)
        
        # Записываем новые значения в массивы путей
        paths_rub[:, t] = r_rub_new
        paths_usd[:, t] = r_usd_new
        paths_fx[:, t]  = S_new
    return paths_rub, paths_usd, paths_fx