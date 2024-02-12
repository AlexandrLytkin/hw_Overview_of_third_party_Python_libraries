def test_requests():  # pip3 install requests
    import requests
    result = requests.get("https://randomuser.me/api/")
    print(result)  # узнать коды состояний
    print(result.url)
    response = result.text
    print(response)


def test_pandas():  # Инструмен для анализа обработки данных, напоминает эксэль
    import numpy as np
    import pandas as pd
    data = ["Pandas", "Matplotlib", "Numpy"]
    s = pd.Series(data, index=['a', 'b', 'c'])
    print(s)
    s_random = pd.Series(np.random.randn(6), index=['p', 'a', 'n', 'd', 'a', 's'])
    print(s_random)
    print(s.index)
    print(s_random.index)
    my_dict = {'cat': 12, 'dog': 32, 'nail': 100500}
    res = pd.Series(my_dict)
    print(res)

    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    df = pd.DataFrame(data)
    df2 = pd.DataFrame(data, index=['1', '2', '3'], columns=['a', 'b', 'c'])
    data2 = {'name': ['vasya', 'katya', 'masha'], 'age': [23, 32, 12]}
    df3 = pd.DataFrame(data2)
    print(df)
    print(df2)
    print(df3)


def test_matplotlib():  # Отличный инструмент для работ с числами особенно с много мерными матрицами!
    import numpy as np
    data = [1, 2, 3, 4, 5]
    data2 = ([1, 2, 3, 4, 5])
    arr = np.array(data)
    arr2 = np.array(data2)
    arr3 = np.array(data, dtype=float)
    print(arr, arr2, arr3)
    print(arr.shape, arr2.shape, arr3.shape)
    print(arr.dtype, arr2.dtype, arr3.dtype)
    print(arr.ndim, arr2.ndim, arr3.ndim)
    print(len(arr3))
    print(arr3.size)
    arr3 = arr3.astype(np.int64)
    print(arr3.dtype)
    arr4 = np.arange(0, 20, 1.5)
    print(arr4)
    arr5 = np.linspace(0, 2, 5)
    print(arr5)
    random_arr = np.random.random((5,))
    print(random_arr)
    random_arr2 = np.random.randint(-5, 20, 10)
    print(random_arr2)
    matrix = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    print(matrix)
    new_matrix = np.arange(16).reshape(2, 8)
    new_matrix2 = np.arange(16)
    print(new_matrix)
    print(new_matrix2)


def test_matplotlib():  # pip3 install matplotlib Библиотека Matplotlib для построения графиков
    import matplotlib.pyplot as plt
    # x = [1,2,3,4,5]
    # y = [25, 32, 34, 20, 25]
    # plt.xlabel('Ось X')
    # plt.ylabel('Ось Y')
    # plt.title('Первый график')
    # # plt.plot(x, y)
    # plt.plot(x, y, color='green', marker='o', markersize=7)
    # plt.show()
    print('*' * 30)
    x = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май']
    y = [2, 4, 3, 1, 7]
    plt.bar(x, y, label='Величина прибыли')
    plt.xlabel('Месяц года')
    plt.ylabel('Прибыль в млн.руб.')
    plt.title('Пример столбчатой диаграммы')
    plt.legend()
    plt.show()


def test_pillow():  # pip3 install pillow Инструмент для работы с изображением
    from PIL import ImageFilter
    from PIL import Image

    # SOURSE_DIR = 'image/'
    # p1 = Image.open(SOURSE_DIR + 'image1.jpg')
    # print(p1.size)
    # print(p1.mode)
    # print(p1.format)
    # print(p1.info)
    # p1.show()
    # new_image = p1.crop((0, 0, p1.width, p1.width)).resize((300, 300)).transpose(Image.FLIP_LEFT_RIGHT)
    # new_image.save(SOURSE_DIR + 'shreder.jpg')
    # new_image.show()
    # print('*' * 30)
    # size = (128, 128)
    # original = Image.open('image/programmer.jpg')
    # # original.thumbnail(size)
    # # img = original.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # img = original.filter(ImageFilter.EMBOSS)
    # original.save('image/programmer2.jpg')
    # img.save('image/programmer3.jpg')
    # # original.show()
    # # img.show()
    # print(original.format, original.size, original.mode)
    print('*' * 30)
    img1 = Image.open('image/hobgoblin.jpg')
    img2 = Image.open('image/programmer.jpg')
    # img = Image.new("RGBA", (1000, 1000), 'white')
    # img = img1.crop((100,100,900,900))
    # img = img1.filter(ImageFilter.BLUR)
    img = img1.filter(ImageFilter.EDGE_ENHANCE)
    img.save('image/img.png')
    # rotate = img.rotate(180)
    # rotate.show()
    # img.show()


# test_requests()
# test_pandas()
# test_matplotlib()
# test_matplotlib()
# test_pillow()
