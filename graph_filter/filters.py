from dateutil import parser


def perform_filter_on_dataframe(df, filters):
    # print(filters)
    if filters['age']:
        df = df.loc[df['Age'].isin(filters['age'])]

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

        rev = {}
        if (len(filters["date"]) == 1):
            rev['date_in'] = [parser.parse(d) for d in filters["date"][0]]
        else:
            rev['date_'+filters["date"][0]
                ] = [parser.parse(d) for d in filters["date"][1]]

        print("Filtering by date, value= "+str(filters["date"]))
        g = g.subgraph_edges(g.es(**rev))

    if (filters["duree"]):
        duree = {}
        if (len(filters["duree"]) == 1):
            duree['duree_in'] = [str(i) for i in filters["duree"][0]]
        else:
            duree['duree_'+filters["duree"][0]
                  ] = [str(i) for i in filters["duree"][1]]

        print("Filtering by duree, value= "+str(filters["duree"]))
        g = g.subgraph_edges(g.es.select(**duree))

    if (filters["service"]):
        print("Filtering by service, value= "+str(filters["service"]))
        g = g.subgraph_edges(g.es.select(service_in=filters["service"]))

    if (filters["accorderie"]):
        filters['accorderie'] = [int(x) for x in filters['accorderie']]
        print("Filtering by accorderie, value= "+str(filters["accorderie"]))
        g = g.subgraph_edges(g.es.select(accorderie_in=filters["accorderie"]))

    return g

