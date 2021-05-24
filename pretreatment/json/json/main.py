def main(json_value, key):
    if type(json_value) is str:
        json_value = eval(json_value)
    return {"value": json_value.get(key)}

