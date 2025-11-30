import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import statsmodels.api as sm


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


# ======================================
# 3. ОЦЕНКА ВОЛАТИЛЬНОСТИ FX
# ======================================

def estimate_fx_vol_from_series(
    S_series: pd.Series,
    dt_years: float = 1 / 252,
):
    """
    Оценивает годовую волатильность курса по лог-доходностям.

    Parameters
    ----------
    S_series : pd.Series
        Ряд значений курса (S_t).
    dt_years : float
        Шаг по времени между наблюдениями (в годах).

    Returns
    -------
    sigma_annual : float
        Годовая волатильность лог-доходностей.
    sigma_step : float
        Волатильность на один шаг (дневную, если dt=1/252).
    log_returns : np.ndarray
        Массив лог-доходностей.
    """
    S = np.asarray(S_series, dtype=float)
    log_ret = np.diff(np.log(S))
    sigma_step = log_ret.std(ddof=1)
    # годовая вола: масштабируем на sqrt(1/dt)
    sigma_annual = sigma_step * np.sqrt(1.0 / dt_years)
    return float(sigma_annual), float(sigma_step), log_ret


# ======================================
# 4. КОРРЕЛЯЦИОННАЯ МАТРИЦА ДЛЯ (Δr_RUB, Δr_USD, log_ret_FX)
# ======================================

def compute_corr_matrix(
    dr_rub: np.ndarray,
    dr_usd: np.ndarray,
    log_ret_fx: np.ndarray,
):
    """
    Строит корреляционную матрицу 3x3 по приращениям ставок и лог-доходностям FX.

    Parameters
    ----------
    dr_rub : np.ndarray
        Приращения рублевой ставки (r_{t+1} - r_t).
    dr_usd : np.ndarray
        Приращения долларовой ставки.
    log_ret_fx : np.ndarray
        Лог-доходности курса.

    Returns
    -------
    corr_matrix : np.ndarray shape (3, 3)
        Корреляционная матрица для (Δr_RUB, Δr_USD, log_ret_FX).
    """
    data = np.vstack([dr_rub, dr_usd, log_ret_fx])
    corr_matrix = np.corrcoef(data)
    return corr_matrix


def simulate_market_paths(
    N_sim: int,
    N_steps: int,
    T_years: float,
    r0_rub: float,
    r0_usd: float,
    s0_fx: float,
    cir_rub_params: tuple,
    cir_usd_params: tuple,
    fx_vol: float,
    corr_matrix: np.ndarray,
    seed: int = 42,
):
    """
    Симулирует динамику трех риск-факторов:
    - краткосрочная ставка RUB (CIR),
    - краткосрочная ставка USD (CIR),
    - курс USD/RUB (логнормальная модель с дрейфом r_rub - r_usd).

    Параметры
    ---------
    N_sim : int
        Количество сценариев Монте-Карло.
    N_steps : int
        Количество временных шагов на горизонте T_years.
    T_years : float
        Горизонт моделирования в годах.
    r0_rub, r0_usd : float
        Начальные значения краткосрочных ставок.
    s0_fx : float
        Начальное значение курса USD/RUB.
    cir_rub_params : (kappa_r, theta_r, sigma_r)
        Параметры CIR для рублевой ставки.
    cir_usd_params : (kappa_d, theta_d, sigma_d)
        Параметры CIR для долларовой ставки.
    fx_vol : float
        Годовая волатильность логнормальной модели курса.
    corr_matrix : np.ndarray
        3x3 корреляционная матрица для (RUB, USD, FX).
    seed : int
        Seed для генератора случайных чисел (для воспроизводимости).

    Возвращает
    ----------
    rub_rates, usd_rates, fx_rates, discount_integral : np.ndarray
        Массивы траекторий (N_sim, N_steps+1) и интеграл рублевой ставки по времени для каждого сценария.
    """
    dt = T_years / N_steps
    kappa_r, theta_r, sigma_r = cir_rub_params
    kappa_d, theta_d, sigma_d = cir_usd_params

    # Инициализация массивов траекторий
    rub_rates = np.zeros((N_sim, N_steps + 1))
    usd_rates = np.zeros((N_sim, N_steps + 1))
    fx_rates = np.zeros((N_sim, N_steps + 1))

    rub_rates[:, 0] = r0_rub
    usd_rates[:, 0] = r0_usd
    fx_rates[:, 0] = s0_fx

    # Интеграл рублевой ставки (для дисконтирования)
    discount_integral = np.zeros(N_sim)

    # Разложение Холецкого корреляционной матрицы
    L = np.linalg.cholesky(corr_matrix)

    # Фиксируем seed для воспроизводимости
    np.random.seed(seed)

    for t in range(N_steps):
        # Генерация независимых стандартных нормальных
        Z_uncorr = np.random.normal(0, 1, size=(3, N_sim))
        # Ввод корреляций
        Z = L @ Z_uncorr

        r_r = rub_rates[:, t]
        r_d = usd_rates[:, t]
        S_t = fx_rates[:, t]

        # Обновление ставки RUB по CIR
        dr_r = (
            kappa_r * (theta_r - r_r) * dt
            + sigma_r * np.sqrt(np.maximum(r_r, 0)) * np.sqrt(dt) * Z[0, :]
        )
        rub_rates[:, t + 1] = r_r + dr_r

        # Обновление ставки USD по CIR
        dr_d = (
            kappa_d * (theta_d - r_d) * dt
            + sigma_d * np.sqrt(np.maximum(r_d, 0)) * np.sqrt(dt) * Z[1, :]
        )
        usd_rates[:, t + 1] = r_d + dr_d

        # Обновление курса FX (логнормальная модель под риск-нейтральной мерой в RUB)
        # Дрифт = (r_rub - r_usd) - 0.5 * sigma_fx^2
        drift_fx = (r_r - r_d - 0.5 * fx_vol**2) * dt
        diffusion_fx = fx_vol * np.sqrt(dt) * Z[2, :]
        fx_rates[:, t + 1] = S_t * np.exp(drift_fx + diffusion_fx)

        # Накопление интеграла рублевой ставки для дисконтирования
        discount_integral += r_r * dt

    return rub_rates, usd_rates, fx_rates, discount_integral


def price_range_accrual(
    notional: float,
    lower_barrier: float | None,
    upper_barrier: float | None,
    T_years: float,
    r0_rub: float,
    r0_usd: float,
    s0_fx: float,
    cir_rub_params: tuple,
    cir_usd_params: tuple,
    fx_vol: float,
    corr_matrix: np.ndarray,
    N_sim: int = 10_000,
    N_steps: int = 252,
):
    """
    Оценка справедливой стоимости Range Accrual.

    Параметры
    ---------
    notional : float
        Номинал сделки (максимальная выплата при 100% времени в диапазоне).
    lower_barrier, upper_barrier : float или None
        Нижняя и верхняя границы диапазона. Если границы отсутствуют,
        можно передать None.
    T_years : float
        Срок сделки в годах.
    r0_rub, r0_usd : float
        Начальные краткосрочные ставки в рублях и долларах.
    s0_fx : float
        Начальный курс USD/RUB.
    cir_rub_params, cir_usd_params : tuple
        Параметры CIR для RUB и USD.
    fx_vol : float
        Годовая волатильность курса.
    corr_matrix : np.ndarray
        Корреляционная матрица трёх факторов.
    N_sim : int
        Количество симуляций.
    N_steps : int
        Количество временных шагов.

    Возвращает
    ----------
    fair_value : float
        Оценка справедливой стоимости дериватива.
    std_error : float
        Оценка стандартной ошибки Монте-Карло.
    rub_rates, usd_rates, fx_rates : np.ndarray
        Смоделированные траектории для анализа и графиков.
    """
    rub_rates, usd_rates, fx_rates, discount_integral = simulate_market_paths(
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

    # Берем все шаги, кроме начального
    S_paths = fx_rates[:, 1:]  # shape: (N_sim, N_steps)

    # Условие попадания в диапазон
    in_range = np.ones_like(S_paths, dtype=bool)
    if lower_barrier is not None:
        in_range &= S_paths >= lower_barrier
    if upper_barrier is not None:
        in_range &= S_paths <= upper_barrier

    # Доля дней в диапазоне по каждому сценарию
    accrual_fraction = in_range.mean(axis=1)

    # Дисконт-фактор по рублевой ставке
    df = np.exp(-discount_integral)

    # Выплата и приведённая стоимость по каждому сценарию
    payoff = notional * accrual_fraction
    pv_paths = payoff * df

    fair_value = pv_paths.mean()
    std_error = pv_paths.std(ddof=1) / np.sqrt(N_sim)

    return fair_value, std_error, rub_rates, usd_rates, fx_rates