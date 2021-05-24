import run as r


def main(file_name, sheet_name=0):
    return r.run(file_name=file_name,
                 sheet_name=sheet_name)


if __name__ == '__main__':
    import json
    a = main('E:\\mlp_share\\set\\admin\\2\\77939cc0e7ef4da09a9592d7c4ba58c3.xlsx')

    print(a)
    json.dumps(a)
