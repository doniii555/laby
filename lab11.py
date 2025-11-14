def solve_tridiagonal(a, b, c, d):
    n = len(b)
    if not (len(a) == len(c) == len(d) == n):
        raise ValueError("Все массивы a, b, c, d должны быть одинаковой длины")

    # Прямой ход: считаем коэффициенты альфа и бета
    alpha = [0.0] * n
    beta  = [0.0] * n

    # Инициализация для первой строки
    if b[0] == 0:
        raise ZeroDivisionError("b[0] = 0, деление на ноль в первой строке")

    alpha[0] = -c[0] / b[0]
    beta[0]  = d[0] / b[0]

    for i in range(1, n):
        denom = a[i] * alpha[i - 1] + b[i]  
        if denom == 0:
            raise ZeroDivisionError(
                f"Нулевой знаменатель на шаге i={i}. Метод прогонки неприменим (нулевой диагональный элемент)."
            )
        alpha[i] = -c[i] / denom if i < n - 1 else 0 
        beta[i]  = (d[i] - a[i] * beta[i - 1]) / denom

    x = [0.0] * n
    x[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


def input_tridiagonal_system():
    print("Решение трёхдиагональной СЛАУ методом прогонки.")
    n = int(input("Введите размер системы n: "))

    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    d = [0.0] * n

    print("\nВвод коэффициентов для уравнений:")
    print("Формат: a_i b_i c_i d_i")
    print("Важно: для первой строки a_0 = 0, для последней строки c_{n-1} = 0\n")

    for i in range(n):
        print(f"Строка {i} (уравнение {i + 1}):")
        if i == 0:
            row = input("Введите b_0 c_0 d_0 (a_0 = 0): ").split()
            if len(row) != 3:
                raise ValueError("Нужно ввести ровно 3 числа")
            a[i] = 0.0
            b[i] = float(row[0])
            c[i] = float(row[1])
            d[i] = float(row[2])
        elif i == n - 1:
            row = input(f"Введите a_{i} b_{i} d_{i} (c_{i} = 0): ").split()
            if len(row) != 3:
                raise ValueError("Нужно ввести ровно 3 числа")
            a[i] = float(row[0])
            b[i] = float(row[1])
            c[i] = 0.0
            d[i] = float(row[2])
        else:
            row = input(f"Введите a_{i} b_{i} c_{i} d_{i}: ").split()
            if len(row) != 4:
                raise ValueError("Нужно ввести ровно 4 числа")
            a[i] = float(row[0])
            b[i] = float(row[1])
            c[i] = float(row[2])
            d[i] = float(row[3])

    return a, b, c, d


def main():
    while True:
        print("\n=== Калькулятор СЛАУ (метод прогонки) ===")
        print("1. Ввести систему вручную")
        print("2. Пример системы (готовые коэффициенты)")
        print("0. Выход")
        choice = input("Ваш выбор: ").strip()

        if choice == "0":
            print("Выход из программы.")
            break

        if choice == "1":
            a, b, c, d = input_tridiagonal_system()
        elif choice == "2":
            a = [0, 1, 1]
            b = [2, 3, 2]
            c = [1, 1, 0]
            d = [5, 10, 8]
            print("\nИспользуем примерную систему:")
            print("2x0 + 1x1       = 5")
            print("1x0 + 3x1 + 1x2 = 10")
            print("     1x1 + 2x2  = 8")
        else:
            print("Неизвестный пункт меню, попробуйте ещё раз.")
            continue

        try:
            x = solve_tridiagonal(a, b, c, d)
            print("\nРешение системы:")
            for i, xi in enumerate(x):
                print(f"x[{i}] = {xi}")
        except Exception as e:
            print(f"Ошибка при решении: {e}")


if __name__ == "__main__":
    main()
