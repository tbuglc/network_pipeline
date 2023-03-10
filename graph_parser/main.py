from parse_members import parse_members
from parse_transactions import parse_transaction
from igraph import Graph
import argparse


def remove_missing_edge_vertex(transactions, members):
    mask = transactions.iloc[:, :2].isin(members.index)
    filtered_transactions = transactions[mask]

    print(filtered_transactions)


def test_graph(members, transactions):
    try:
        Graph.DataFrame(transactions, directed=True, vertices=members)
        return True
    except ValueError as err:
        print(err)
        return False


def main(input_dir, output_dir):
    members = parse_members(input_dir)

    print('Input directory: ' + input_dir)
    print('Output directory: ' + output_dir)

    print('Loaded members')

    # print(members)

    transactions = parse_transaction(input_dir ,members=members)
    # print(transactions)
    print('Loaded transactions')

    is_graph = test_graph(members=members, transactions=transactions)
    print('constructing a graph test ')

    if is_graph:
        print('Saving to file')
        members.to_csv(output_dir + '/members.csv', sep=",",
                       encoding="latin-1", index=False)
        transactions.to_csv(output_dir + '/transactions.csv', sep=",",
                            encoding="latin-1", index=False)
    else:
        print("Could not create graph")



arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-i', '--input', required=True)
arg_parser.add_argument('-o', '--output', required=True)

args = arg_parser.parse_args()        
print(args.__dict__)

main(args.input, args.output)
