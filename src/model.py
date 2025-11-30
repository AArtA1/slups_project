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
# 1. ГРАФИК ИСТОРИЧЕСКИХ ДАННЫХ ПО DATAFRAME
# ======================================

def plot_history_df(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "Дата",
    figsize=(12, 5),
    date_range=None,
    y_limits=None,
    y_margin: float = 0.05,
    rolling_window: int | None = None,
    show_points: bool = False,
):
    """
    Рисует красивый график временного ряда (по двум колонкам DataFrame).

    Parameters
    ----------
    df : pd.DataFrame
        Таблица с данными.
    time_col : str
        Название колонки с датами.
    value_col : str
        Название колонки со значениями.
    title : str
        Заголовок графика.
    ylabel : str
        Подпись по оси Y.
    xlabel : str
        Подпись по оси X.
    figsize : tuple
        Размер фигуры (ширина, высота).
    date_range : (start, end) или None
        Ограничение по датам. Можно строки 'YYYY-MM-DD' или datetime.
    y_limits : (y_min, y_max) или None
        Фиксированные границы по оси Y.
    y_margin : float
        Запас по оси Y (в долях диапазона), если y_limits не заданы.
    rolling_window : int или None
        Если задано, поверх рисуется скользящее среднее с таким окном.
    show_points : bool
        Рисовать ли маркеры точек поверх линии.
    """
    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(time_col)

    if date_range is not None:
        start, end = date_range
        if start is not None:
            start = pd.to_datetime(start)
            data = data[data[time_col] >= start]
        if end is not None:
            end = pd.to_datetime(end)
            data = data[data[time_col] <= end]

    dates = data[time_col].values
    values = data[value_col].astype(float).values

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dates, values, linewidth=2, label=value_col)

    if show_points:
        ax.scatter(dates, values, s=10, alpha=0.7)

    if rolling_window is not None and rolling_window > 1:
        roll = data[value_col].rolling(rolling_window, min_periods=1).mean().values
        ax.plot(dates, roll, color="black", linestyle="--",
                linewidth=1.5, label=f"MA({rolling_window})")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.grid(True, which="major", linestyle="--", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        y_min = np.nanmin(values)
        y_max = np.nanmax(values)
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        else:
            pad = (y_max - y_min) * y_margin
            y_min -= pad
            y_max += pad
        ax.set_ylim(y_min, y_max)

    if rolling_window is not None and rolling_window > 1:
        ax.legend()

    plt.tight_layout()
    plt.show()


# ======================================
# 2. КАЛИБРОВКА CIR ПО РЯДУ СТАВОК
# ======================================

def calibrate_cir_from_series(
    r_series: pd.Series,
    dt_years: float = 1 / 252,
):
    """
    Калибрует параметры CIR-модели по ряду краткосрочных ставок.

    Используется Эйлеровская дискретизация:
    dr_t = kappa (theta - r_t) dt + sigma * sqrt(r_t) * dW_t,
    и линейная регрессия вида:
    (r_{t+1} - r_t)/sqrt(r_t) = A * (1/sqrt(r_t)) + B * sqrt(r_t) + шум,
    где A = kappa * theta * dt, B = -kappa * dt.

    Parameters
    ----------
    r_series : pd.Series
        Ряд ставок (в ДОЛЯХ, не в процентах) с равномерным шагом dt.
    dt_years : float
        Длина временного шага в годах (для дневных данных обычно 1/252).

    Returns
    -------
    kappa : float
        Скорость возврата к среднему.
    theta : float
        Долгосрочный средний уровень ставки.
    sigma : float
        Волатильность краткосрочной ставки.
    model : statsmodels.regression.linear_model.RegressionResults
        Объект с результатами OLS-регрессии (можно смотреть summary()).
    """
    r = np.asarray(r_series, dtype=float)
    r_t = r[:-1]
    r_next = r[1:]

    # фильтруем слишком маленькие/отрицательные значения
    mask = r_t > 1e-8
    r_t = r_t[mask]
    r_next = r_next[mask]

    y = (r_next - r_t) / np.sqrt(r_t)
    x1 = 1.0 / np.sqrt(r_t)
    x2 = np.sqrt(r_t)

    X = pd.DataFrame({"x1": x1, "x2": x2})

    model = sm.OLS(y, X).fit()

    A = model.params["x1"]
    B = model.params["x2"]

    kappa = -B / dt_years
    theta = A / (kappa * dt_years)
    sigma = model.resid.std(ddof=1) / np.sqrt(dt_years)

    return float(kappa), float(theta), float(sigma), model


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


# ======================================
# 5. СИМУЛЯЦИЯ ТРЁХ РИСК-ФАКТОРОВ
# ======================================

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
):
    """
    Симулирует траектории ставок RUB, USD и курса USD/RUB.

    Модели:
    - r_RUB: CIR
    - r_USD: CIR
    - S: логнормальная модель под риск-нейтральной мерой в RUB: дрейф = r_RUB - r_USD.

    Parameters
    ----------
    N_sim : int
        Количество сценариев Монте-Карло.
    N_steps : int
        Количество шагов по времени.
    T_years : float
        Горизонт моделирования в годах.
    r0_rub, r0_usd : float
        Начальные краткосрочные ставки (в долях).
    s0_fx : float
        Начальный курс USD/RUB.
    cir_rub_params : (kappa_r, theta_r, sigma_r)
        Параметры CIR для RUB.
    cir_usd_params : (kappa_d, theta_d, sigma_d)
        Параметры CIR для USD.
    fx_vol : float
        Годовая волатильность курса (логнормальной модели).
    corr_matrix : np.ndarray shape (3, 3)
        Корреляционная матрица для шумов (RUB, USD, FX).

    Returns
    -------
    rub_rates : np.ndarray shape (N_steps+1, N_sim)
    usd_rates : np.ndarray shape (N_steps+1, N_sim)
    fx_rates  : np.ndarray shape (N_steps+1, N_sim)
    discount_integrals : np.ndarray shape (N_sim,)
        Интеграл ставки RUB по времени (для дисконтирования).
    """
    dt_step = T_years / N_steps
    kappa_r, theta_r, sigma_r = cir_rub_params
    kappa_d, theta_d, sigma_d = cir_usd_params

    rub = np.zeros((N_steps + 1, N_sim))
    usd = np.zeros((N_steps + 1, N_sim))
    fx = np.zeros((N_steps + 1, N_sim))
    disc_int = np.zeros(N_sim)

    rub[0, :] = r0_rub
    usd[0, :] = r0_usd
    fx[0, :] = s0_fx

    L = np.linalg.cholesky(np.asarray(corr_matrix, dtype=float))

    for t in range(N_steps):
        Z = np.random.normal(size=(3, N_sim))
        dW = np.sqrt(dt_step) * (L @ Z)

        r_rub = rub[t, :]
        r_usd = usd[t, :]
        s = fx[t, :]

        # CIR для RUB
        dr_rub = (
            kappa_r * (theta_r - r_rub) * dt_step
            + sigma_r * np.sqrt(np.maximum(r_rub, 0.0)) * dW[0, :]
        )

        # CIR для USD
        dr_usd = (
            kappa_d * (theta_d - r_usd) * dt_step
            + sigma_d * np.sqrt(np.maximum(r_usd, 0.0)) * dW[1, :]
        )

        # FX под риск-нейтральной мерой в RUB: дрейф = r_rub - r_usd
        dlnS = ((r_rub - r_usd - 0.5 * fx_vol**2) * dt_step
                + fx_vol * dW[2, :])

        rub[t + 1, :] = np.maximum(r_rub + dr_rub, 0.0)
        usd[t + 1, :] = np.maximum(r_usd + dr_usd, 0.0)
        fx[t + 1, :] = s * np.exp(dlnS)

        disc_int += r_rub * dt_step

    return rub, usd, fx, disc_int


# ======================================
# 6. ПРАЙСИНГ RANGE ACCRUAL
# ======================================

def price_range_accrual(
    notional: float,
    lower_barrier: float | None,
    upper_barrier: float | None,
    start_date: dt.date,
    end_date: dt.date,
    r0_rub: float,
    r0_usd: float,
    s0_fx: float,
    cir_rub_params: tuple,
    cir_usd_params: tuple,
    fx_vol: float,
    corr_matrix: np.ndarray,
    N_sim: int = 10000,
    steps_per_year: int = 252,
):
    """
    Оценивает справедливую стоимость Range Accrual.

    Платёж = notional * (доля времени, когда S_t в диапазоне [L, U]),
    дисконтирование по краткосрочной ставке в RUB (CIR).

    lower_barrier и upper_barrier могут быть None (нет нижней / верхней границы).

    Parameters
    ----------
    notional : float
        Номинал сделки (максимальная возможная выплата).
    lower_barrier : float или None
        Нижняя граница диапазона.
    upper_barrier : float или None
        Верхняя граница диапазона.
    start_date, end_date : datetime.date
        Даты начала и окончания контракта.
    r0_rub, r0_usd, s0_fx : float
        Начальные значения ставок и курса на дату начала.
    cir_rub_params, cir_usd_params : tuple
        Параметры CIR для RUB и USD.
    fx_vol : float
        Годовая волатильность курса.
    corr_matrix : np.ndarray
        Корреляционная матрица.
    N_sim : int
        Количество сценариев.
    steps_per_year : int
        Число шагов в году (обычно 252).

    Returns
    -------
    fair_value : float
        Оценка справедливой стоимости.
    std_error : float
        Стандартная ошибка оценки Монте-Карло.
    """
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

    S_paths = fx[1:, :]  # исключаем начальный момент
    in_range = np.ones_like(S_paths, dtype=bool)
    if lower_barrier is not None:
        in_range &= S_paths >= lower_barrier
    if upper_barrier is not None:
        in_range &= S_paths <= upper_barrier

    frac_in_range = in_range.mean(axis=0)

    discount_factor = np.exp(-disc_int)
    payoff = notional * frac_in_range
    pv_scenarios = payoff * discount_factor

    fair_value = pv_scenarios.mean()
    std_error = pv_scenarios.std(ddof=1) / np.sqrt(N_sim)

    return float(fair_value), float(std_error)


# ======================================
# 7. ВСПОМОГАТЕЛЬНЫЕ ГРАФИКИ ДЛЯ СИМУЛЯЦИЙ
# ======================================

def plot_sample_paths(
    time_grid: np.ndarray,
    paths: np.ndarray,
    title: str = "",
    ylabel: str = "",
    n_paths: int = 20,
):
    """
    Рисует несколько сэмплов траекторий из массива paths.

    Parameters
    ----------
    time_grid : np.ndarray shape (N_steps+1,)
        Временная сетка (в годах).
    paths : np.ndarray shape (N_steps+1, N_sim)
        Массив смоделированных траекторий.
    title : str
        Заголовок графика.
    ylabel : str
        Подпись по оси Y.
    n_paths : int
        Сколько траекторий рисовать.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    n_plot = min(n_paths, paths.shape[1])
    ax.plot(time_grid, paths[:, :n_plot])
    ax.set_title(title)
    ax.set_xlabel("Время, годы")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_terminal_hist(
    paths: np.ndarray,
    title: str = "",
    xlabel: str = "",
    bins: int = 40,
):
    """
    Рисует гистограмму конечных значений смоделированного процесса.

    Parameters
    ----------
    paths : np.ndarray shape (N_steps+1, N_sim)
        Смоделированные траектории.
    title : str
        Заголовок.
    xlabel : str
        Подпись оси X.
    bins : int
        Количество бинов в гистограмме.
    """
    terminal = paths[-1, :]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(terminal, bins=bins, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Плотность")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_mc_convergence(
    N_list: list,
    price_func,
):
    """
    Рисует график сходимости оценки Монте-Карло
    для произвольной функции прайсинга.

    Parameters
    ----------
    N_list : list[int]
        Список чисел симуляций, по которым будет считаться цена.
    price_func : callable
        Функция, которая принимает N_sim и возвращает (fair_value, std_error).
        Например: lambda N: price_range_accrual(..., N_sim=N)
    """
    fair_values = []
    std_errors = []

    for N in N_list:
        fair, se = price_func(N)
        fair_values.append(fair)
        std_errors.append(se)
        print(f"N={N}: fair={fair:.4f}, se={se:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(N_list, fair_values, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Число симуляций (лог-шкала)")
    ax.set_ylabel("Оценка fair value")
    ax.set_title("Сходимость Монте-Карло")
    ax.grid(True)
    plt.tight_layout()
    plt.show()