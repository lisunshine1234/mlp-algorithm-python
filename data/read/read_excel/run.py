import pandas as pd

def run(file_name, sheet_name):
    array = pd.read_excel(file_name, sheet_name=sheet_name, header=None)
    return {"array": array.values.tolist()}

