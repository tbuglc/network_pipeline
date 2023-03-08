import pandas as pd
def create_xlsx_file(file_name: str):
    if (file_name == ''):
        raise ValueError('Missing file name')

    return pd.ExcelWriter(
        file_name+'.xlsx', engine='xlsxwriter')


def add_sheet_to_xlsx(file_writer=pd.ExcelWriter, data=pd.DataFrame, title='', index=False):
    data.to_excel(
        file_writer, sheet_name=title, index=index)


def save_csv_file(file_writer=pd.ExcelWriter):
    file_writer.save()

