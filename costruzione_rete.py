import json
from igraph import Graph
from igraph import plot

with open("muslim/counter/4ca1286c-6cfd-11ee-a31f-0242ac170005/channel_video_and_comments.json") as f:
    d = json.load(f)
vids = d["channel_video_and_comments"]["videos"]
d_comm = {}
for video in vids:
    id = video["videoId"]
    d_comm[id] = []
    comments = video["comments"]["comments"]
    replies = video["comments"]["replies"]
    for comment in comments:
        d_comm[id].append(comment["Author"])
    for reply in replies:
        d_comm[id].append(reply["Author"])

ids =[]
for e in d_comm:
    ids += list(set(d_comm[e]))

def ids_in_n(n, ids):
    ids_in_n = []
    for id in ids:
        c=0
        for k in d_comm:
            if id in d_comm[k]:
                c += 1
                if c == n:
                    ids_in_n.append(id)
                    break
    return ids_in_n

lneighs = []
dic = {id: n for n, id in enumerate(ids)}

for id in ids:
    for k in d_comm:
        comms = d_comm[k]
        if id in comms:
            for id1 in comms:
                if id1 != id:
                    lneighs.append((dic[id], dic[id1]))
        else:
            continue


net = Graph()
net.add_vertices(len(ids))
net.add_edges(lneighs)
layout = net.layout_kamada_kawai()
plot(net, layout=layout, bbox=(1600, 1600)).save("dd.png")  # Adjust the 'bbox' dimensions as needed
