import numpy as np
from matplotlib import pyplot as plt


def average_embedding(M_embedd):
    avg = M_embedd.mean(0)  # mean row
    return avg


def average_embedding_n(M_embedd):
    avg = M_embedd.mean(0)  # mean row
    avg = avg / np.linalg.norm(avg)
    return avg


def create_axis(vec_1, vec_2):
    return vec_2 - vec_1


def create_axis_n(vec_1, vec_2):
    x = vec_2 - vec_1
    return x / np.linalg.norm(x)


# histogramm functions
def split_by_attribute(values, meta, attribute: str):
    pairsort(meta, values, attribute)
    legend = []
    split_data = []
    i = 0
    i_old = 0
    while i < len(values):
        s = meta[i]
        legend.append(s)

        while (i < len(values)) and (meta[i] == s):
            i += 1

        split_data.append(values[i_old:i])
        i_old = i
    return split_data, legend


def pairsort(meta, embedds, attribute: str):
    pairt = [(meta[i][attribute], embedds[i]) for i in range(0, len(meta))]
    pairt.sort()

    for i in range(0, len(embedds)):
        meta[i] = pairt[i][0]
        embedds[i] = pairt[i][1]


def show_hist(values, title):
    print("starting histogramm")
    # Plotting a basic histogram
    plt.hist(values, bins=30, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title)

    x_min, x_max = plt.xlim()
    max_abs = max(abs(x_max), abs(x_min))
    plt.xlim(-max_abs, max_abs)

    # Display the plot
    plt.show()


def get_average_attribute(num_bins: int, values, meta, attribute: str):
    # pairsort values, meta by values
    pairt = [(values[i], meta[i][attribute]) for i in range(0, len(values))]
    pairt.sort()

    for i in range(0, len(values)):
        values[i] = pairt[i][0]
        meta[i] = pairt[i][1]

    # print(values)
    # calculate avarage for each bin
    width = (values[len(values) - 1] - values[0]) / num_bins
    print("width: ", width)
    stop = values[0] + width
    avgs = []
    i = 0  # iterator
    c = 0  # number of entries in bin
    sum = 0  # sum over current bin

    while i < len(values):
        # print(stop, values[i])
        while values[i] < stop or i == len(values) - 1:
            sum += meta[i]
            c += 1
            i += 1
            if i == len(values):
                break

        avgs.append(sum / c) if c != 0 else avgs.append(0)
        c = 0
        sum = 0
        stop += width

    return avgs
    # adjust last sum


def show_stacked_hist(values, meta, attribute, num_bins, title):
    # avgs = get_average_attribute(num_bins, values, meta.copy(), attribute="wls")
    # print("avgs: ", avgs)

    split_data, legend = split_by_attribute(values, meta.copy(), attribute)
    # print(legend)
    # print(split_data)
    plt.hist(split_data, bins=num_bins, stacked=True, edgecolor='black')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title)

    # center axis without messing up scale
    x_min, x_max = plt.xlim()
    max_abs = max(abs(x_max), abs(x_min))
    plt.xlim(-max_abs, max_abs)

    # Adding legend
    plt.legend(legend)

    # try showing average wls for every bin
    """
    try:
        w= 0.5/num_bins
        bin_centers = [(i*w-0.25) for i in range(0,num_bins)]
        for avg, bin_center in zip(avgs, bin_centers):
            plt.annotate(text=str(int(avg)), xy=(bin_center, 0.5))
    except Exception as error:
        print(error)
    """
    # Display the plot
    plt.show()
