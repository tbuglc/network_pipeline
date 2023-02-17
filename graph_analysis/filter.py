

from graph_loader import load_accorderie_network
import main


def perform_filter(gender='male'):

    # process args
    # call filter with processed args

    g = load_accorderie_network()
    filtered_node = g.vs.select(genre=gender)

    sub_graph = g.induced_subgraph(filtered_node)

    trx = sub_graph.get_edge_dataframe()
    members = sub_graph.get_vertex_dataframe()

    # trx.set_index('edge ID', inplace=True, drop=True)
    # members.set_index('name', inplace=True. drop)

    # generate folder name
    folder_name = 'filtered_data'
    # call main to calculate metrics
    # main(span_days=30, folder_name='filtered_data_metrics/', g=sub_graph)
    # # export data

    # t_file_writer = create_xlsx_file('./artifacts/'+folder_name+'transactions')
    # add_sheet_to_xlsx(file_writer=t_file_writer, data=trx, title='trx')
    # save_csv_file(file_writer=t_file_writer)

    # m_file_writer = create_xlsx_file('./artifacts/'+folder_name+'members')
    # add_sheet_to_xlsx(file_writer=m_file_writer, data=members, title='m')
    # save_csv_file(file_writer=m_file_writer)

    return trx, members
