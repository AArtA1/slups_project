import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


np.random.seed(42)


def plot_history(series, title, ylabel):
    """
    Строит график исторического временного ряда (ставка или курс валюты).

    Функция создаёт линейный график временного ряда с датами на оси X.
    Используется для визуализации исторических данных ставок (RUONIA, SOFR)
    или обменных курсов (USD/RUB).

    Parameters
    ----------
    series : pd.Series
        Временной ряд с индексом типа datetime. Содержит значения ставки
        или курса для каждой даты.
    title : str
        Заголовок графика.
    ylabel : str
        Подпись по оси Y (например, "Ставка, доли" или "Курс USD/RUB").

    Returns
    -------
    None
        Функция отображает график через matplotlib.pyplot.show().

    Examples
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2019-01-01', periods=100, freq='D')
    >>> rates = pd.Series(np.random.randn(100).cumsum() + 0.05, index=dates)
    >>> plot_history(rates, "RUONIA", "Ставка, доли")
    """
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values)
    ax.set_title(title)
    ax.set_xlabel("Дата")
    ax.set_ylabel(ylabel)
    plt.show()


def ols_cir(data, dt):
    """
    Оценивает параметры модели CIR (Cox-Ingersoll-Ross) методом наименьших квадратов.

    Модель CIR описывает динамику краткосрочной процентной ставки:
    dr_t = κ(θ - r_t)dt + σ√(r_t)dW_t

    Функция использует дискретизацию модели в виде AR(1) процесса и применяет
    линейную регрессию для оценки параметров κ (kappa), θ (theta) и σ (sigma).

    Parameters
    ----------
    data : np.ndarray
        Одномерный массив исторических значений ставки (в долях, не в процентах).
        Должен содержать последовательные наблюдения во времени.
    dt : float
        Шаг по времени между наблюдениями в годах (например, 1/252 для дневных данных).

    Returns
    -------
    k0 : float
        Оценка параметра κ (kappa) - скорость возврата к долгосрочному уровню.
        Положительное значение означает, что ставка стремится к theta.
    theta0 : float
        Оценка параметра θ (theta) - долгосрочное среднее значение ставки.
        Должно быть положительным.
    sigma0 : float
        Оценка параметра σ (sigma) - волатильность ставки.
        Вычисляется как стандартное отклонение остатков регрессии, масштабированное на √dt.

    Notes
    -----
    Метод основан на преобразовании дискретной формы CIR модели:
    (r_{t+1} - r_t) / √(r_t) = κθ * dt / √(r_t) - κ * dt * √(r_t) + σ * √(dt) * ε_t

    Где ε_t - стандартная нормальная случайная величина.

    Examples
    --------
    >>> import numpy as np
    >>> # Пример: дневные данные ставки за год (252 торговых дня)
    >>> rates = np.array([0.05, 0.051, 0.049, 0.052, ...])  # в долях
    >>> kappa, theta, sigma = ols_cir(rates, dt=1/252)
    >>> print(f"kappa={kappa:.4f}, theta={theta:.4f}, sigma={sigma:.4f}")
    """

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


def estimate_fx_vol_from_series(
    S_series: pd.Series,
    dt_years: float = 1 / 252,
):
    """
    Оценивает годовую волатильность обменного курса по историческим лог-доходностям.

    Функция вычисляет стандартное отклонение лог-доходностей и масштабирует его
    для получения годовой волатильности. Используется для калибровки параметра
    волатильности в логнормальной модели обменного курса.

    Parameters
    ----------
    S_series : pd.Series
        Временной ряд значений обменного курса (например, USD/RUB).
        Индекс должен быть типа datetime. Значения должны быть положительными.
    dt_years : float, optional
        Шаг по времени между наблюдениями в годах. По умолчанию 1/252
        (один торговый день).

    Returns
    -------
    sigma_annual : float
        Годовая волатильность лог-доходностей. Вычисляется как:
        sigma_annual = sigma_step * √(1/dt_years)
    sigma_step : float
        Волатильность на один шаг (дневная волатильность, если dt=1/252).
        Вычисляется как стандартное отклонение лог-доходностей с поправкой
        Бесселя (ddof=1).
    log_returns : np.ndarray
        Массив лог-доходностей: log(S_{t+1}/S_t) = log(S_{t+1}) - log(S_t).

    Notes
    -----
    Лог-доходности вычисляются как: log_ret_t = ln(S_{t+1}) - ln(S_t)

    Масштабирование на √(1/dt) основано на свойстве броуновского движения:
    если волатильность на шаг dt равна σ_step, то годовая волатильность
    равна σ_step * √(252) для дневных данных.

    Examples
    --------
    >>> import pandas as pd
    >>> # Загрузка данных курса USD/RUB
    >>> fx_data = pd.read_csv("usdrub_data.csv", index_col="Date", parse_dates=True)
    >>> sigma_ann, sigma_daily, log_rets = estimate_fx_vol_from_series(
    ...     fx_data["RUB=X"], dt_years=1/252
    ... )
    >>> print(f"Годовая волатильность: {sigma_ann:.4f}")
    """
    S = np.asarray(S_series, dtype=float)
    log_ret = np.diff(np.log(S))
    sigma_step = log_ret.std(ddof=1)
    # годовая вола: масштабируем на sqrt(1/dt)
    sigma_annual = sigma_step * np.sqrt(1.0 / dt_years)
    return float(sigma_annual), float(sigma_step), log_ret


def compute_corr_matrix(
    dr_rub: np.ndarray,
    dr_usd: np.ndarray,
    log_ret_fx: np.ndarray,
):
    """
    Вычисляет корреляционную матрицу 3x3 для трёх риск-факторов.

    Функция строит корреляционную матрицу между:
    - приращениями рублевой ставки (Δr_RUB),
    - приращениями долларовой ставки (Δr_USD),
    - лог-доходностями обменного курса (log_ret_FX).

    Эта матрица используется для моделирования совместной динамики трёх факторов
    в симуляциях Монте-Карло с учётом корреляций между ними.

    Parameters
    ----------
    dr_rub : np.ndarray
        Одномерный массив приращений рублевой ставки: r_{t+1} - r_t.
        Длина массива должна совпадать с dr_usd и log_ret_fx.
    dr_usd : np.ndarray
        Одномерный массив приращений долларовой ставки: r_{t+1} - r_t.
    log_ret_fx : np.ndarray
        Одномерный массив лог-доходностей обменного курса: ln(S_{t+1}/S_t).

    Returns
    -------
    corr_matrix : np.ndarray, shape (3, 3)
        Симметричная корреляционная матрица 3x3. Элементы:
        - corr_matrix[0, 0] = 1.0 (корреляция Δr_RUB с собой)
        - corr_matrix[0, 1] = корреляция между Δr_RUB и Δr_USD
        - corr_matrix[0, 2] = корреляция между Δr_RUB и log_ret_FX
        - corr_matrix[1, 2] = корреляция между Δr_USD и log_ret_FX
        И так далее (матрица симметрична).

    Notes
    -----
    Матрица должна быть положительно определённой для использования в разложении
    Холецкого. Функция использует numpy.corrcoef для вычисления выборочных
    корреляций Пирсона.

    Examples
    --------
    >>> import numpy as np
    >>> # Вычисление приращений ставок
    >>> dr_rub = np.diff(rub_rates)  # приращения RUONIA
    >>> dr_usd = np.diff(usd_rates)  # приращения SOFR
    >>> log_ret_fx = np.diff(np.log(fx_rates))  # лог-доходности USD/RUB
    >>> corr = compute_corr_matrix(dr_rub, dr_usd, log_ret_fx)
    >>> print(f"Корреляция RUB-USD: {corr[0, 1]:.4f}")
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
    Симулирует совместную динамику трёх скоррелированных риск-факторов методом Монте-Карло.

    Функция генерирует N_sim траекторий для трёх факторов:
    1. Краткосрочная ставка RUB (модель CIR)
    2. Краткосрочная ставка USD (модель CIR)
    3. Обменный курс USD/RUB (логнормальная модель)

    Корреляции между факторами моделируются через разложение Холецкого
    корреляционной матрицы.

    Parameters
    ----------
    N_sim : int
        Количество сценариев (путей) Монте-Карло. Рекомендуется >= 10,000
        для получения стабильных оценок.
    N_steps : int
        Количество временных шагов на горизонте T_years. Обычно 252 для
        дневных данных (один торговый год).
    T_years : float
        Горизонт моделирования в годах (например, 1.0 для одного года).
    r0_rub : float
        Начальное значение краткосрочной ставки в рублях (в долях, не процентах).
        Например, 0.075 для 7.5%.
    r0_usd : float
        Начальное значение краткосрочной ставки в долларах (в долях).
    s0_fx : float
        Начальное значение обменного курса USD/RUB (например, 75.0).
    cir_rub_params : tuple of float, shape (3,)
        Параметры CIR модели для рублевой ставки: (kappa_r, theta_r, sigma_r).
        - kappa_r: скорость возврата к долгосрочному уровню
        - theta_r: долгосрочное среднее значение ставки
        - sigma_r: волатильность ставки
    cir_usd_params : tuple of float, shape (3,)
        Параметры CIR модели для долларовой ставки: (kappa_d, theta_d, sigma_d).
    fx_vol : float
        Годовая волатильность логнормальной модели обменного курса.
        Должна быть положительной.
    corr_matrix : np.ndarray, shape (3, 3)
        Корреляционная матрица для трёх факторов в порядке:
        [RUB ставка, USD ставка, FX курс].
        Матрица должна быть симметричной и положительно определённой.
    seed : int, optional
        Seed для генератора случайных чисел. По умолчанию 42.
        Используется для обеспечения воспроизводимости результатов.

    Returns
    -------
    rub_rates : np.ndarray, shape (N_sim, N_steps + 1)
        Массив траекторий рублевой ставки. Каждая строка - один сценарий,
        каждый столбец - значение ставки в момент времени.
    usd_rates : np.ndarray, shape (N_sim, N_steps + 1)
        Массив траекторий долларовой ставки.
    fx_rates : np.ndarray, shape (N_sim, N_steps + 1)
        Массив траекторий обменного курса USD/RUB.
    discount_integral : np.ndarray, shape (N_sim,)
        Интеграл рублевой ставки по времени для каждого сценария:
        ∫₀ᵀ r_rub(t) dt. Используется для вычисления дисконт-фактора
        exp(-discount_integral) при оценке стоимости деривативов.

    Notes
    -----
    Моделирование выполняется в дискретной форме:

    1. **CIR модель для ставок:**
       dr_t = κ(θ - r_t)dt + σ√(r_t)dW_t
       Дискретизация: r_{t+1} = r_t + κ(θ - r_t)dt + σ√(r_t)√(dt)Z_t

    2. **Логнормальная модель для курса:**
       d ln(S_t) = (r_rub - r_usd - 0.5*σ²_FX)dt + σ_FX dW_t
       Дискретизация: S_{t+1} = S_t * exp((r_rub - r_usd - 0.5*σ²_FX)dt + σ_FX√(dt)Z_t)

    Корреляции вводятся через разложение Холецкого: Z = L @ Z_uncorr,
    где L - нижняя треугольная матрица из разложения Холецкого корреляционной матрицы.

    Examples
    --------
    >>> import numpy as np
    >>> # Параметры CIR для RUB и USD
    >>> cir_rub = (1.52, 0.058, 0.092)  # kappa, theta, sigma
    >>> cir_usd = (1.55, 0.002, 0.165)
    >>> # Корреляционная матрица
    >>> corr = np.array([[1.0, 0.038, 0.010],
    ...                  [0.038, 1.0, -0.018],
    ...                  [0.010, -0.018, 1.0]])
    >>> # Симуляция
    >>> rub, usd, fx, discount = simulate_market_paths(
    ...     N_sim=10000, N_steps=252, T_years=1.0,
    ...     r0_rub=0.075, r0_usd=0.001, s0_fx=75.0,
    ...     cir_rub_params=cir_rub, cir_usd_params=cir_usd,
    ...     fx_vol=0.129, corr_matrix=corr, seed=42
    ... )
    >>> print(f"Средняя финальная ставка RUB: {rub[:, -1].mean():.4f}")
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
    Оценивает справедливую стоимость Range Accrual дериватива методом Монте-Карло.

    Range Accrual — дериватив, выплата по которому пропорциональна доле времени,
    в течение которого обменный курс находился в заданном диапазоне [lower_barrier, upper_barrier].

    Выплата в момент T: Payoff = Notional * (количество дней в диапазоне / общее количество дней)

    Справедливая стоимость вычисляется как математическое ожидание дисконтированной выплаты
    под риск-нейтральной мерой, где дисконтирование выполняется по рублевой ставке.

    Parameters
    ----------
    notional : float
        Номинал сделки в рублях. Это максимальная возможная выплата, которая
        достигается, если курс находился в диапазоне 100% времени.
    lower_barrier : float or None
        Нижняя граница диапазона для обменного курса USD/RUB.
        Если None, нижняя граница не учитывается (только верхняя).
    upper_barrier : float or None
        Верхняя граница диапазона для обменного курса USD/RUB.
        Если None, верхняя граница не учитывается (только нижняя).
        Если оба барьера None, выплата всегда равна номиналу.
    T_years : float
        Срок сделки в годах (например, 1.0 для одного года).
    r0_rub : float
        Начальная краткосрочная ставка в рублях (в долях, не процентах).
    r0_usd : float
        Начальная краткосрочная ставка в долларах (в долях).
    s0_fx : float
        Начальный обменный курс USD/RUB на дату оценки.
    cir_rub_params : tuple of float, shape (3,)
        Параметры CIR модели для рублевой ставки: (kappa_r, theta_r, sigma_r).
    cir_usd_params : tuple of float, shape (3,)
        Параметры CIR модели для долларовой ставки: (kappa_d, theta_d, sigma_d).
    fx_vol : float
        Годовая волатильность обменного курса (положительное число).
    corr_matrix : np.ndarray, shape (3, 3)
        Корреляционная матрица для трёх риск-факторов (RUB ставка, USD ставка, FX курс).
    N_sim : int, optional
        Количество симуляций Монте-Карло. По умолчанию 10,000.
        Увеличение N_sim улучшает точность, но увеличивает время вычислений.
    N_steps : int, optional
        Количество временных шагов на горизонте T_years. По умолчанию 252
        (дневные шаги для одного года). Должно быть достаточно большим для
        точного моделирования непрерывных процессов.

    Returns
    -------
    fair_value : float
        Оценка справедливой стоимости Range Accrual дериватива в рублях.
        Вычисляется как среднее дисконтированных выплат по всем сценариям:
        E[exp(-∫r_rub dt) * Payoff]
    std_error : float
        Стандартная ошибка оценки Монте-Карло. Вычисляется как:
        std_error = std(pv_paths) / √(N_sim)
        Используется для построения доверительных интервалов.
    rub_rates : np.ndarray, shape (N_sim, N_steps + 1)
        Массив траекторий рублевой ставки для всех сценариев.
        Может использоваться для анализа и построения графиков.
    usd_rates : np.ndarray, shape (N_sim, N_steps + 1)
        Массив траекторий долларовой ставки.
    fx_rates : np.ndarray, shape (N_sim, N_steps + 1)
        Массив траекторий обменного курса USD/RUB.
        Используется для определения попадания в диапазон барьеров.

    Notes
    -----
    Алгоритм оценки:
    1. Генерируются N_sim траекторий трёх риск-факторов через simulate_market_paths().
    2. Для каждого сценария вычисляется доля времени, когда курс был в диапазоне.
    3. Выплата = notional * (доля времени в диапазоне).
    4. Приведённая стоимость = выплата * exp(-∫r_rub dt).
    5. Справедливая стоимость = среднее приведённых стоимостей по всем сценариям.

    Дисконтирование выполняется по рублевой ставке, так как выплата номинирована в рублях.

    Examples
    --------
    >>> import numpy as np
    >>> # Параметры сделки
    >>> notional = 1_000_000  # 1 млн рублей
    >>> lower_barrier = 70.0
    >>> upper_barrier = 80.0
    >>> # Параметры моделей
    >>> cir_rub = (1.52, 0.058, 0.092)
    >>> cir_usd = (1.55, 0.002, 0.165)
    >>> corr = np.array([[1.0, 0.038, 0.010],
    ...                  [0.038, 1.0, -0.018],
    ...                  [0.010, -0.018, 1.0]])
    >>> # Оценка стоимости
    >>> fv, se, rub, usd, fx = price_range_accrual(
    ...     notional=notional,
    ...     lower_barrier=lower_barrier,
    ...     upper_barrier=upper_barrier,
    ...     T_years=1.0,
    ...     r0_rub=0.075, r0_usd=0.001, s0_fx=75.0,
    ...     cir_rub_params=cir_rub,
    ...     cir_usd_params=cir_usd,
    ...     fx_vol=0.129,
    ...     corr_matrix=corr,
    ...     N_sim=10000,
    ...     N_steps=252
    ... )
    >>> print(f"Справедливая стоимость: {fv:,.2f} руб.")
    >>> print(f"Стандартная ошибка: {se:,.2f} руб.")
    >>> print(f"95% доверительный интервал: [{fv - 1.96*se:,.2f}, {fv + 1.96*se:,.2f}]")
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