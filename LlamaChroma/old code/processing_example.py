import data_example as de


def horrible_threshhold_query(query_text: str, content: str, threshold: float):
    n = 10
    temp = de.query_hp(query_text, content, n)

    while temp["distances"][0][n - 1] < threshold:
        n += 10
        temp = de.query_hp(query_text, content, n)

    while temp["distances"][0][n - 1] > threshold:
        for atr in temp.values():
            # print(atr)
            if atr is not None:
                atr[0].pop()
        n -= 1
        if n == 0:
            break

    return temp

def avg_distance(results):
    distances = results["distances"][0]
    return sum(distances)/len(distances)

def avg_position(results):
    ids = [int(i) for i in results["ids"][0]]
    return sum(ids)/len(ids)
    distances = results["distances"][0]
    return len(distances)

def result_length(results):
    distances = results["distances"][0]
    return len(distances)



if __name__ == '__main__':
    query_list = ["magic", "school", "friends", "finance", "neurology", "family", "scary", "Star Wars","Hogwarts","night", "Harry Potter","Bob"]
    contains = " "
    stats_list = []
    print("query_text   ", "length  ", "distances   ", "position")

    for i in range(len(query_list)):
        query_text = query_list[i]
        results = horrible_threshhold_query(query_text=query_text, content=contains, threshold=1.5)

        length = result_length(results)
        distances = avg_distance(results)
        position = avg_position(results)/de.hp_collection.count()
        temp = [query_text, length,distances, position]
        stats_list.append(temp)
        # de.print_query_results(results=results)  # id, distance, text

        print(query_text, length, distances, position)

    print("query_text   ", "length  ", "distances   ", "position")
    for i in stats_list: print(i)


