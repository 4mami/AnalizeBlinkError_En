from SPARQLWrapper import SPARQLWrapper, JSON
import tqdm

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
tmp_query = """
SELECT DISTINCT ?categories
WHERE {
?categories skos:broader+ <http://dbpedia.org/resource/Category:Hidden_categories> .
}
ORDER BY ?categories
LIMIT 1000
"""

with open("outputs/query_result.txt", "a") as f:
    for i in tqdm.tqdm(range(40)):
        if (i > 0):
            query = tmp_query + "OFFSET " + str(i) + "000"
        else:
            query = tmp_query

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if not results["results"]["bindings"]:
            break

        for result in results["results"]["bindings"]:
            f.write(result["categories"]["value"] + "\n")