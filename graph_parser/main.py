
from parse_members import parse_members
from parse_transactions import parse_transaction
from igraph import Graph

output_dir = '.\output\\'

def remove_missing_edge_vertex(transactions, members):
    mask = transactions.iloc[:,:2].isin(members.index)
    filtered_transactions = transactions[mask]

    print(filtered_transactions)

def test_graph(members, transactions):
    try:
        Graph.DataFrame(transactions, directed=True, vertices=members)
        return True
    except ValueError as err:
        print(err)
        return False


def main():
    members = parse_members()
    print('Loaded members')

    # print(members)

    transactions = parse_transaction(members=members)
    # print(transactions)
    print('Loaded transactions')

    is_graph = test_graph(members=members, transactions=transactions)
    print('constructing a graph test ')

    if is_graph:
        print('Saving to file')
        members.to_csv(output_dir+'members.csv', sep=",",
                       encoding="latin-1", index=False)
        transactions.to_csv(output_dir+'transactions.csv', sep=",",
                            encoding="latin-1", index=False)
    else:
        print("Could not create graph")


main()
