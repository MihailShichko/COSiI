import numpy as np
from matplotlib import pyplot as plt

N = 8 # число дискретных отсчетов
T = 2 * np.pi

# двоичная перестановка элементов массива по коду Грея (двоичная инверсия)
def bin_reorder(y):
    size = len(y)                                                # количество элементов массива
    num_bits = len(bin(size - 1)[2:])                            # получаем max длину двоичного представления индекса

    reordered_y = np.zeros(size, dtype=np.complex64)            # массив для результата

    for i in range(size):
        bin_index = bin(i)[2:].zfill(num_bits)                   # получаем двоичное значение индекса и дополняем 0 до max длины индекса
        inverted_bin_index = bin_index[::-1]                     # переворачиваем двоичное значение индекса
        decimal_index = int(inverted_bin_index, 2)               # переводим в двоичную систему счисления

        if decimal_index <= size - 1:                            # если новый индекс не больше размера массива -> производим перестановку
            reordered_y[decimal_index] = y[i]                    # записываем по новому индексу текущее значение
    
    return reordered_y

# алгоритм БПФ с прореживанием по частоте
def dif_fft(y, size, direction):

    if size == 1:                                                                
        return y
    
    # вычисляем главный комплексный корень N-й степени из 1
    wN = complex(np.cos(2 * np.pi / size), direction * np.sin(2 * np.pi / size))
    w = 1

    # пустые массивы для первой и второй половин масссива
    b = np.zeros(size // 2, dtype=np.complex64)
    c = np.zeros(size // 2, dtype=np.complex64)

    # бабочка бабочка бабочка моя
    for j in range(size // 2):
        b[j] = y[j] + y[j + size // 2]
        c[j] = (y[j] - y[j + size // 2]) * w
        w *= wN

    # рекурсивно вызываем БПФ на каждой из частей
    y_b = dif_fft(b, size // 2, direction)
    y_c = dif_fft(c, size // 2, direction)

    # объединяем результаты
    y = np.concatenate((y_b,y_c))

    return y

def linear_convolution(a, b):
    len_a = len(a)
    len_b = len(b)
    result = np.zeros(len_a + len_b - 1)

    a = np.array(a)
    b = np.array(b)

    a = np.pad(a,(0,(len_a + len_b - 1) - len_a),'constant')
    b = np.pad(b,(0,(len_a + len_b - 1) - len_b),'constant')

    for n in range(len_a + len_b - 1):
        for m in range(0, n + 1):
            result[n] += a[m] * b[n - m]

    return result

def cyclic_convolution(a, b):
    size = len(a)
    
    # Создаем пустой массив для результата свертки
    result = np.zeros(size)

    a = np.array(a)
    b = np.array(b)
    b = b[::-1]
    index = 0
    
    # Выполняем циклическую свертку
    for n in range(size):
        b = np.roll(b,1)
        index += 1
        for m in range(index) :            
            result[n] += a[m] * b[m]
    
    return result

# исходный сигнал
def signal_y(x):
    return np.cos(x)

def signal_z(x):
    return np.sin(x)

def build_graphs(y,z,lin_conv,cycl_conv,correlation,fft_conv,lib_corr,lib_conv,x,double_x):
    fig = plt.figure()
    fig.canvas.manager.set_window_title('LAB2')

    func_y = fig.add_subplot(7,1,1)
    func_y.plot(x, y, color='green')
    func_y.set_xlabel('x')
    func_y.set_ylabel('y')
    func_y.set_title('Исходная функция Y')

    func_z = fig.add_subplot(7,1,2)
    func_z.plot(x, z, color='blue')
    func_z.set_xlabel('x')
    func_z.set_ylabel('y')
    func_z.set_title('Исходная функция Z')

    lin_conv_func = fig.add_subplot(8,1,3)
    lin_conv_func.plot(double_x, lin_conv, color='red')
    lin_conv_func.set_xlabel('x')
    lin_conv_func.set_ylabel('y')
    lin_conv_func.set_title('Линейная свертка')

    cycl_conv_func = fig.add_subplot(8,1,4)
    cycl_conv_func.plot(x, cycl_conv, color='grey')
    cycl_conv_func.set_xlabel('x')
    cycl_conv_func.set_ylabel('y')
    cycl_conv_func.set_title('Циклическая свертка')

    lib_conv_func = fig.add_subplot(8,1,5)
    lib_conv_func.plot(double_x, lib_conv, color='black')
    lib_conv_func.set_xlabel('x')
    lib_conv_func.set_ylabel('y')
    lib_conv_func.set_title('Библиотечная свертка')

    corr_func = fig.add_subplot(8,1,6)
    corr_func.plot(x, correlation, color='brown')
    corr_func.set_xlabel('x')
    corr_func.set_ylabel('y')
    corr_func.set_title('Корреляция')

    lib_corr_func = fig.add_subplot(8,1,7)
    lib_corr_func.plot(x, lib_corr, color='purple')
    lib_corr_func.set_xlabel('x')
    lib_corr_func.set_ylabel('y')
    lib_corr_func.set_title('Библиотечная корреляция')

    fft_conv_func = fig.add_subplot(8,1,8)
    fft_conv_func.plot(x, fft_conv, color='pink')
    fft_conv_func.set_xlabel('x')
    fft_conv_func.set_ylabel('y')
    fft_conv_func.set_title('Свертка БПФ')

    fig.subplots_adjust(hspace=1.5)
    fig.show()

def main():
    x = np.linspace(0.0, N - 1, N)
    double_x = np.linspace(0, 2 * N - 2, 2 * N - 1)

    y = signal_y(x)
    z = signal_z(x)

    lin_conv = linear_convolution(y,z)
    cycl_conv = cyclic_convolution(y,z)

    y_fft = dif_fft(y,N,1)
    y_fft = bin_reorder(y_fft)

    z_fft = dif_fft(z,N,1)
    z_fft = bin_reorder(z_fft)

    correlation = y_fft * z_fft
    correlation = dif_fft(correlation,N,-1) / N
    correlation = bin_reorder(correlation)

    y_fft_conjugated = y_fft.conjugate()

    fft_conv = y_fft_conjugated * z_fft
    fft_conv = dif_fft(fft_conv,N,-1) / N
    fft_conv = bin_reorder(fft_conv)


    lib_corr = np.correlate(y,z,'full')[:N][::-1]

    lib_conv = np.convolve(y,z,'full')

    build_graphs(y,z,lin_conv,cycl_conv,correlation,fft_conv,lib_corr,lib_conv,x,double_x)
    input()

if __name__ == "__main__":
    main()
