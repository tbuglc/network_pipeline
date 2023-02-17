import pandas as pd
from datetime import datetime, date
import numpy as np

input_dir = '.\input\\'


def load_data():
    transactions = pd.read_csv(input_dir+'echange_services.csv',
                               sep=";", encoding="latin-1")

    categorie = pd.read_csv(
        input_dir+'categorie.csv', sep=";", encoding="latin-1")

    sous_category = pd.read_csv(
        input_dir+'categorie_sous_categorie.csv', sep=";", encoding="latin-1")

    offre_service = pd.read_csv(
        input_dir+'offre_services.csv', sep=";", encoding="latin-1")

    print('INIT transaction shape: ' + str(transactions.shape))

    return transactions, categorie, sous_category, offre_service


def populate_columns_data():
    transactions, categorie, sous_category, offre_service = load_data()
    '''
        T
        # transactions = transactions[["NoEchangeService", "NoMembreVendeur", "NoMembreAcheteur", "NbHeure",
        #                              "DateEchange", "NoOffreServiceMembre"]]

        # of_s = offre_service[["NoOffreServiceMembre",
        #                       "NoCategorieSousCategorie", "NoAccorderie"]]

        # transactions['NoOffreServiceMembre'] = pd.to_numeric(
        #     transactions['NoOffreServiceMembre'], errors='coerce')

    '''
    transactions['NoOffreServiceMembre'] = pd.to_numeric(
        transactions['NoOffreServiceMembre'], errors='coerce', downcast='integer')

    print('Type casting, new shape: ' + str(transactions.shape))

    transactions = transactions.merge(
        offre_service, on="NoOffreServiceMembre", how="left")

    print('Joined with OFFRE SERVICE, new transactions shape : ' +
          str(transactions.shape))

    transactions = transactions.merge(
        sous_category, on="NoCategorieSousCategorie", how="left")
    print('Joined with CATEGORIE SOUS CATEGORIE, new transactions shape : ' +
          str(transactions.shape))

    transactions = transactions.merge(categorie, on="NoCategorie", how="left")
    print('Joined with CATEGORIE, new transactions shape : ' +
          str(transactions.shape))

    return transactions


def drop_trx_with_nan_vendeur_acheteur(transactions):
    print('BEFORE DROPING transaction shape: ' + str(transactions.shape))

    transactions['source'].replace('', np.nan, inplace=True)
    transactions['target'].replace('', np.nan, inplace=True)

    transactions = transactions.dropna(subset=['source', 'target'])
    print('AFTER DROPING transaction shape: ' + str(transactions.shape))

    return transactions


def format_transactions(transactions):
    print('Pruning columns, new shape: ' + str(transactions.shape))

    transactions = transactions[["NoEchangeService", "NoMembreVendeur",
                                 "NoMembreAcheteur", "NbHeure", "DateEchange", "TitreCategorie", "TitreOffre", "NoAccorderie"]]

    transactions = transactions.rename(columns={"NoEchangeService": "id", "NoMembreVendeur": "source", "NoMembreAcheteur": "target",
                                                "DateEchange": "date", "TitreCategorie": "service", "Sexe": "genre", "NbHeure": "duree", "TitreOffre": "detailservice", "NoAccorderie": "accorderie"})

    print('Renamed columns')

    transactions = transactions[["source", "target", "duree",
                                "date", "service",  "detailservice",  "accorderie", "id"]]

    return transactions


def converting(x):
    try:
        x = int(x)
    except ValueError as err:
        x = ''
    return x


def map_transactions_members(members, transactions):

    id_to_idx_map = dict(zip(members["mapid"], members["id"]))

    transactions["source"] = transactions["source"].map(
        id_to_idx_map, na_action='ignore').apply(converting)
    transactions["target"] = transactions["target"].map(
        id_to_idx_map, na_action='ignore').apply(converting)

    transactions = drop_trx_with_nan_vendeur_acheteur(
        transactions=transactions)

    transactions['source'] = transactions['source'].astype(int)

    transactions['target'] = transactions['target'].astype(int)

    return transactions


def parse_transaction(members):
    transactions = populate_columns_data()
    transactions = format_transactions(transactions=transactions)

    transactions = drop_trx_with_nan_vendeur_acheteur(
        transactions=transactions)
    transactions = map_transactions_members(
        members=members, transactions=transactions)

    return transactions
