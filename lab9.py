import numpy as np

def solve_dufort_frankel(L, T, N, M, a_squared, initial_condition, boundary_condition_L, boundary_condition_R):
    """
    Решает одномерное уравнение теплопроводности u_t = a^2 * u_xx 
    с использованием абсолютно устойчивой схемы Дюфорта-Франкля (схема ромб)[cite: 41].

    Схема: (u_i^(j+1) - u_i^(j-1)) / (2*tau) = (u_{i+1}^j - (u_i^(j+1) + u_i^(j-1)) + u_{i-1}^j) / h^2 [cite: 42]

    :param L: Область по x (от 0 до L).
    :param T: Время счета (от 0 до T).
    :param N: Количество интервалов по x (N+1 узлов).
    :param M: Количество слоев по t (M+1 слоев).
    :param a_squared: Коэффициент теплопроводности (a^2).
    :param initial_condition: Функция для начального условия u(x, 0).
    :param boundary_condition_L: Функция для краевого условия u(0, t).
    :param boundary_condition_R: Функция для краевого условия u(L, t).
    :return: Матрица решений u[j][i], где j - слой по времени, i - узел по пространству.
    """
    
    h = L / N            # Шаг по пространству
    tau = T / M          # Шаг по времени
    sigma = (a_squared * tau) / (h * h)
    
    # Инициализация массива решения (M+1 слоев, N+1 узел)
    u = np.zeros((M + 1, N + 1))

    print(f"Параметры сетки: h = {h:.4f}, tau = {tau:.6f}, sigma = {sigma:.4f}")
    
    # 1. Применение начальных условий (j=0) 
    x_nodes = np.linspace(0, L, N + 1)
    u[0, :] = initial_condition(x_nodes)
    u[0, 0] = boundary_condition_L(0)
    u[0, N] = boundary_condition_R(0)

    # 2. Расчет первого слоя (j=1) с помощью Явной схемы (для старта трехслойной схемы)
    # u_i^1 = u_i^0 + sigma * (u_{i+1}^0 - 2*u_i^0 + u_{i-1}^0)
    for i in range(1, N):
        u[1, i] = u[0, i] + sigma * (u[0, i + 1] - 2 * u[0, i] + u[0, i - 1])
    
    # Краевые условия для j=1
    u[1, 0] = boundary_condition_L(tau)
    u[1, N] = boundary_condition_R(tau)


    # 3. Расчет остальных слоев (j=2 до M) с помощью Схемы Дюфорта-Франкля
    # u_i^(j+1) * (1 + 2*sigma) = 2*sigma*(u_{i+1}^j + u_{i-1}^j) + u_i^(j-1) * (1 - 2*sigma)
    
    # Перепишем схему для удобства: u_i^(j+1) = ( 2*sigma*(u_{i+1}^j + u_{i-1}^j) + u_i^(j-1) * (1 - 2*sigma) ) / (1 + 2*sigma)
    D = 1.0 + 2.0 * sigma  # Знаменатель
    E = 1.0 - 2.0 * sigma  # Коэффициент при u_i^(j-1)

    for j in range(1, M):
        # Внутренние узлы (i от 1 до N-1)
        # В Python индексация срезов [1:N] берет узлы от 1 до N-1
        u[j + 1, 1:N] = (2.0 * sigma * (u[j, 2:N+1] + u[j, 0:N-1]) + u[j - 1, 1:N] * E) / D
        
        # Краевые условия
        u[j + 1, 0] = boundary_condition_L((j + 1) * tau)
        u[j + 1, N] = boundary_condition_R((j + 1) * tau)

    return u

# --- Пример использования ---
if __name__ == '__main__':
    # Параметры задачи
    L = 1.0     # Длина
    T = 0.1     # Время
    N = 20      # Узлов по x (количество интервалов)
    M = 1000    # Слоев по t (количество интервалов)
    A2 = 1.0    # Коэффициент теплопроводности a^2 = 1.0

    # Начальное условие: u(x, 0) = sin(pi * x)
    def initial_condition(x):
        return np.sin(np.pi * x)

    # Краевые условия (Дирихле): u(0, t) = 0, u(L, t) = 0
    def boundary_condition_L(t):
        return 0.0

    def boundary_condition_R(t):
        return 0.0

    # Запуск решения
    result_matrix = solve_dufort_frankel(
        L, T, N, M, A2, 
        initial_condition, 
        boundary_condition_L, 
        boundary_condition_R
    )

    # Вывод результатов для последнего слоя (t=T)
    print(f"\n--- Результат (t = {T}) ---")
    x_values = np.linspace(0, L, N + 1)
    
    # Вывод каждого второго узла
    for i in range(0, N + 1, 2):
        print(f"x={x_values[i]:.3f}, u={result_matrix[M, i]:.6f}")
