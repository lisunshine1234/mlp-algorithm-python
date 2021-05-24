import run as r

def main(array, test_size=None, train_size=None, random_state=None, shuffle=None):
    if type(array) is str:
        array = eval(array)
    if type(test_size) is str:
        test_size = eval(test_size)
    if type(train_size) is str:
        train_size = eval(train_size)
    if type(random_state) is str:
        random_state = eval(random_state)
    if type(shuffle) is str:
        shuffle = eval(shuffle)
    return r.run(array, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=shuffle)



if __name__ == '__main__':
    import numpy as np
    import json
    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    back = main(array)


    print(back)
    for i in back:
        print(i + ":" + str(back[i]))
    import json

    json.dumps(back)