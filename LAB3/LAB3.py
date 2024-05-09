import numpy as np
import matplotlib.pyplot as plt


N = 1024 # число дискретных отсчетов
T = 2 * np.pi

# исходный сигнал
def signal(x):
    return np.sin(2*x) + np.cos(7*x)


#Окно Блэкмана
def blackman_window(numtaps):
    n = np.arange(numtaps)
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (numtaps - 1)) + 0.08 * np.cos(4 * np.pi * n / (numtaps - 1))
    return window


def firwin_reject(numtaps, cutoff, width):
    taps = np.sinc(2 * cutoff * (np.arange(numtaps) - (numtaps - 1) / 2))
    window = blackman_window(numtaps)
    taps *= window

    reject = np.sinc(2 * width * (np.arange(numtaps) - (numtaps - 1) / 2))
    rejectWindow = blackman_window(numtaps)
    reject *= rejectWindow

    filter_taps = (taps - reject) * -1
    #filter_taps = (taps - reject)
    return filter_taps

def main():
    x = np.linspace(0, 2*np.pi, N)
    originalSignal = signal(x) #исходный график

    noiseSignal = originalSignal + np.cos(100*x)

    count = 64 #длина фильтра
    cutoff = 5 / count #центральная частота полосы режекции
    width = 0.1 * cutoff #ширина полосы режекции

    filteredSignal = np.convolve(noiseSignal, firwin_reject(count, cutoff, width), mode='same')

    filter = (np.fft.fft(firwin_reject(count, cutoff, width)))

    plt.figure(figsize=(12, 6))

    plt.subplot(4, 2, 1)
    plt.plot(x, originalSignal)
    plt.title('Исходный график')

    plt.subplot(4, 2, 2)
    plt.stem(range(1024), np.fft.fft(originalSignal))
    plt.title('Исходный график АЧХ')

    plt.subplot(4, 2, 3)
    plt.plot(x, noiseSignal)
    plt.title('График с шумом')

    plt.subplot(4, 2, 4)
    plt.stem(range(1024), np.fft.fft(noiseSignal))
    plt.title('График с шумом АЧХ')

    plt.subplot(4, 2, 5)
    plt.plot(x, filteredSignal)
    plt.title('График после использования КИХ фильтра')

    plt.subplot(4, 2, 6)
    plt.stem(range(1024), np.fft.fft(filteredSignal))
    plt.title('График после использования КИХ фильтра АЧХ')

    plt.subplot(4, 2, 8)
    plt.plot(np.abs(filter))
    plt.title('АЧХ')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()