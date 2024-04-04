import numpy as np
from matplotlib import pyplot as plt
import math

N = 64 # число дискретных отсчетов
T = 2 * np.pi

# исходный сигнал
def signal(x):
    return np.sin(2*x) + np.cos(7*x)

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

def build_graphs(x, y, res, rev_res, name):

    fig = plt.figure()
    fig.canvas.manager.set_window_title(name)

    # исходная функция
    orig_func = fig.add_subplot(4,1,1)
    orig_func.plot(x, y, color='green')
    orig_func.set_xlabel('x')
    orig_func.set_ylabel('y')
    orig_func.set_title('Исходная функция')
    orig_func.scatter(x,y, color = 'red', s=10)
    orig_func.axhline(0,color='black')
    orig_func.axvline(0,color='black')

    # АЧХ БПФ
    fft_afc = fig.add_subplot(4,1,2)
    fft_afc.plot(x, np.abs(res), color='blue')
    fft_afc.set_xlabel('частота')
    fft_afc.set_ylabel('амплитуда')
    fft_afc.set_title('БПФ-АЧХ')
    fft_afc.axhline(0,color='black')
    fft_afc.axvline(0,color='black')

    # ФЧХ БПФ
    fft_pfc = fig.add_subplot(4,1,3)
    fft_pfc.plot(np.real(res), np.imag(res),color='red')
    fft_pfc.set_xlabel('частота')
    fft_pfc.set_ylabel('фаза')
    fft_pfc.set_title('БПФ-ФЧХ')
    fft_pfc.axhline(0,color='black')
    fft_pfc.axvline(0,color='black')

    ifft_func = fig.add_subplot(4,1,4)
    ifft_func.plot(x,np.real(rev_res),color='green')
    ifft_func.set_xlabel('x')
    ifft_func.set_ylabel('y')
    ifft_func.set_title('ОБПФ')
    ifft_func.axhline(0,color='black')
    ifft_func.axvline(0,color='black')

    fig.subplots_adjust(hspace=1.5)
    fig.show()

def main():

    x = np.linspace(0.0,T,N)

    y = signal(x)
    
    res1 = dif_fft(y, N, 1)
    res1 = bin_reorder(res1)

    res2 = np.fft.fft(y, N)

    rev_res1 = dif_fft(res1,N,-1) / N
    rev_res1 = bin_reorder(rev_res1)

    rev_res2 = np.fft.ifft(res2,N)

    build_graphs(x,y,res1,rev_res1,'Our FFT')
    build_graphs(x,y,res2,rev_res2,'Lib FFT')
    input()

if __name__ == "__main__":
    main()