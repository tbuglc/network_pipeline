from dateutil import parser
from igraph import Graph


def perform_filter_on_dataframe(df, filters):
    # print(filters)
    if filters['age']:
        df = df.loc[df['Age'].isin(filters['age'])]

    if filters['genre']:
        df = df.loc[df['Genre'].isin(filters['genre'])]

    if filters['revenu']:
        df = df.loc[df['Revenu'].isin(filters['revenu'])]

    if filters['ville']:
        df = df.loc[df['Ville'].isin(filters['ville'])]

    if filters['arrondissement']:
        df = df.loc[df['Arrondissement'].isin(filters['arrondissement'])]

    if filters['region']:
        df = df.loc[df['Region'].isin(filters['region'])]

    return df


def perform_filter_on_graph(g, filters):
    if (not filters):
        return None

    # vertices properties filter
    if (filters["age"]):
        print("Filtering by age, value= "+str(filters["age"]))
        g = g.induced_subgraph(g.vs.select(age_in=filters["age"]))

    if (filters["adresse"]):
        print("Filtering by adresse, value= "+str(filters["adresse"]))
        g = g.induced_subgraph(g.vs.select(adresse_in=filters["adresse"]))

    if (filters["arrondissement"]):
        print("Filtering by arrondissement, value= " +
              str(filters["arrondissement"]))
        g = g.induced_subgraph(g.vs.select(
            arrondissement_in=filters["arrondissement"]))

        if filters.get("accorderie_node", None):
            accs = [int(ac) for ac in filters['accorderie_node']]
            print("Filtering by member accorderie id, value= " +
                  str(filters["accorderie_node"]))
            g = g.induced_subgraph(g.vs.select(accorderie_in=accs))

    if (filters["ville"]):
        print("Filtering by ville, value= "+str(filters["ville"]))
        g = g.induced_subgraph(g.vs.select(aville_in=filters["ville"]))

    if (filters["genre"]):
        print("Filtering by genre, value= "+str(filters["genre"]))
        g = g.induced_subgraph(g.vs.select(
            genre_in=[int(j) for j in filters["genre"]]))

    if (filters["revenu"]):
        rev = {''+filters["revenu"][0]: filters["revenu"][1]}
        print("Filtering by revenu, value= "+str(filters["revenu"]))
        g = g.induced_subgraph(g.vs(**rev))
    # edges properties
    if (filters["date"]):
        print("Filtering by date, value= "+str(filters["date"]))

        date_filter = filters['date']
        if date_filter[0] == '<':
            g = g.subgraph_edges(g.es.select(
                lambda e: False if e['date'] == '0000-00-00' else parser.parse(e['date']) <= parser.parse(date_filter[1:])))
        elif date_filter[0] == '>':
            g = g.subgraph_edges(g.es.select(
                lambda e: False if e['date'] == '0000-00-00' else parser.parse(e['date']) >= parser.parse(date_filter[1:])))
        elif date_filter[0] == ':':
            date_intervals = date_filter[1:].split(',')
            g = g.subgraph_edges(g.es.select(lambda e: False if e['date'] == '0000-00-00' else parser.parse(
                e['date']) >= parser.parse(date_intervals[0]) and parser.parse(e['date']) <= parser.parse(date_intervals[1])))
        else:
            g = g.subgraph_edges(g.es.select(date_in=[date_filter]))

    if (filters["duree"]):
        for i, d in enumerate(filters['duree']):
            splitted_d = d.split(':')
            if len(splitted_d[0]) < 2:
                filters['duree'][i] = '0' + filters['duree'][i]

        print("Filtering by duree, value= "+str(filters["duree"]))
        g = g.subgraph_edges(g.es.select(duree_in=filters['duree']))

    if (filters["service"]):
        print("Filtering by service, value= "+str(filters["service"]))
        g = g.subgraph_edges(g.es.select(service_in=filters["service"]))

    if (filters["accorderie_edge"]):
        filters["accorderie_edge"] = [int(x)
                                      for x in filters["accorderie_edge"]]
        print("Filtering by accorderie, value= " +
              str(filters["accorderie_edge"]))
        g = g.subgraph_edges(g.es.select(
            accorderie_in=filters["accorderie_edge"]))

    return g
