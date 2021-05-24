def main(array, label_index, remove=False):
    json = {}
    if type(label_index) is str:
        label_index = eval(label_index)
    if type(array) is str:
        array = eval(array)
    if type(remove) is str:
        remove = eval(remove)
    print(array)

    for row in array:
        print(row)
        label = row[label_index]
        if remove:
            row.pop(label_index)
        if not json.__contains__(label):
            json[label] = []
        json[label].append(row)

    return {"json": json}


if __name__ == '__main__':
    import numpy as np
    import json

    array = np.loadtxt('D:\\123_2.csv', delimiter=',')
    array = array[0:20, :]

    y = array[:, -1].tolist()
    x = np.delete(array, -1, axis=1).tolist()
    array = array.tolist()
    array = [[58.3756518169628, 35.12557155589695, 0.0], [-52.822076379863255, 54.56381283212648, 1.0], [41.98330359327853, -35.451613252134266, 0.0],
             [-25.880227010452575, -1.8306642804185114, 3.0], [108.40230811327947, 22.37103842557366, 0.0], [7.414063051501375, 35.49405653782901, 0.0],
             [-26.697682678104712, 13.408596446813235, 0.0], [-29.267564081124302, -0.09527588304113735, 1.0], [47.53442149474482, -50.363912731651936, 0.0],
             [-22.956782253908948, -3.59146135883938, 0.0], [-74.7778239238551, -35.55325757657011, 3.0], [-26.871933042179066, -5.643685018652532, 0.0],
             [-0.9698637611265749, 8.94781098321446, 3.0], [-22.55097695138045, -30.46342237292752, 0.0], [40.564585245619824, 7.173608789317012, 0.0],
             [-33.04748205987291, 19.387589202013423, 1.0], [-29.20843089684892, 26.916210959368897, 0.0], [13.056393483868382, -25.452285823605116, 1.0],
             [35.976549059143444, -1.3025758809466044, 1.0], [-8.256432819681857, -33.640141553366085, 0.0]]
    back = main(array, '-1',remove=True)

    print(back)
    for i in back:
        print(i + ":" + str(back[i]))

    json.dumps(back)
