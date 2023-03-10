import pandas as pd
from datetime import datetime, date
import numpy as np

input_dir = 'data/raw/'
# output_dir = 'data/global_network/'


def anonymize_age(year):
    try:
        year = int(year)
        today = date.today()

        age = today.year - year

        # print('age: '+str(age))

        if (age == today.year):
            return 0

        if (age >= 18 and age <= 30):
            return "18-30"

        if (age >= 31 and age <= 54):
            return "31-54"

        if (age >= 55 and age <= 65):
            return "55-65"

        if (age >= 66 and age <= 80):
            return "66-80"

        if (age >= 81 and age <= 95):
            return "81-95"

        if (age >= 96):
            return "96-150"
    except:
        return 0


def splice_postal_code(code):
    if (code is not None):
        return code[0:3]
    return code


def load_data(input_dir):

    try:
        members = pd.read_csv(input_dir + 'membres.csv', sep=";",
                            encoding="latin-1", low_memory=False)
        revenu = pd.read_csv(
            input_dir + 'revenu_familiale_annuel.csv', sep=";", encoding="latin-1")

        arrondissement = pd.read_csv(
            input_dir + 'arrondissement.csv', sep=";", encoding="latin-1")

        ville = pd.read_csv(input_dir + 'ville.csv', sep=";", encoding="latin-1")
        region = pd.read_csv(input_dir + 'region.csv', sep=";", encoding="latin-1")

        print('INIT Members shape: ' + str(members.shape))

        return members, revenu, arrondissement, ville, region
    except Exception as e:
        print('Something went wront while importing files')
        print(e)


def populate_columns_data(input_dir):
    members,  revenu, arrondissement, ville, region = load_data(input_dir=input_dir+'/')

    ville = ville.merge(region, on="NoRegion", how="left")
    print('Joined with REGION, new member shape : ' + str(members.shape))

    members = members.merge(revenu, on="NoRevenuFamilial", how="left")
    print('Joined with REVENU, new member shape : ' + str(members.shape))

    members = members.merge(arrondissement[[
                            "NoArrondissement", "Arrondissement"]], on="NoArrondissement", how="left")

    print('Joined with ARRONDISSEMENT, new member shape : ' + str(members.shape))

    members = members.merge(ville, on="NoVille", how="left")

    print('Joined with VILLE, new member shape : ' + str(members.shape))

    members = members[["NoMembre", "AnneeNaissance", "CodePostal",
                       "Ville", "Region", "Arrondissement", "Revenu", "Sexe", "NoAccorderie"]]

    return members


def format_members(members):
    print('Pruning columns, new shape: ' + str(members.shape))

    members = members.rename(columns={"NoMembre": "mapid", "AnneeNaissance": "age", "NoAccorderie": "accorderie", "Ville": "ville", "Region": "region", "Arrondissement": "arrondissement",
                                      "Revenu": "revenu", "Sexe": "genre", "CodePostal": "adresse"})
    print('Renamed columns')

    members["age"] = members["age"].apply(anonymize_age)
    print('Age anonymized')
    members["adresse"] = members["adresse"].apply(splice_postal_code)
    print('Address anonymized')

    members['id'] = [id for id in range(len(members['mapid']))]
    print('mapped db id to new indices columns')

    members = members[["id", "age", "genre", "revenu",  "ville",
                       "region", "arrondissement", "adresse", "accorderie", "mapid"]]

    return members


def parse_members(input_dir):
    members = populate_columns_data(input_dir)

    members = format_members(members=members)

    return members
