import run as r


def main(file_name, key):
    return r.run(file_name=file_name,
                 key=key)


if __name__ == '__main__':
    import json
    a = main('E:\\mlp_share\\set\\admin\\2\\75be1ef24b1e42c1ae6eeb9b396db58b.mat', 'Data')

    print(a)
    json.dumps(a)
