
from datetime import date
from dateutil import parser
from graph_loader import data_loader


def filter_by_trans_date(edge, start, end):
    # y, m, d = edge['date'].split('-')
    trans_date = parser.parse(edge['date'])
    # print(trans_date)
    # date(int(y), int(m), int(d))
    if (start <= trans_date and trans_date <= end):
        return True
    return False


def get_start_and_end_date():
    _, trx = data_loader()

    return parser.parse(trx['date'].min()), parser.parse(trx['date'].max())
