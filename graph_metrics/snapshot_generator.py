from datetime import timedelta
from graph_common.utils import filter_by_trans_date


def create_snapshots(g, start_date, end_date, span_days):
    new_date_limit = start_date + timedelta(days=span_days)

    snapshots = []
    while (new_date_limit <= end_date):
        # filter from start to new_date_limit
        edge_list = g.es.select(lambda edge: filter_by_trans_date(
            edge, start_date, new_date_limit))

        sub = None

        if (len(edge_list.indices) != 0):
            sub = g.subgraph_edges(edge_list)

        if (sub is not None):
            snapshots.append({"subgraph": sub, "title": start_date.strftime(
                '%Y-%m-%d') + ' to ' + new_date_limit.strftime('%Y-%m-%d')})

        start_date = new_date_limit
        new_date_limit = new_date_limit + timedelta(span_days)

    return snapshots
