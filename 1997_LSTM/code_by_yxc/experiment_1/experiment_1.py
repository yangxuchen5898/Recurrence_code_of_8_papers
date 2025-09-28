import random
graph={
    "B1":["T1","P1"],
    "T1":["B2"],
    "P1":["B2"],
    "B2":["T2","P2"],
    "T2":["S1","X2"],
    "P2":["T3","V1"],
    "T3":["T3","V1"],
    "S1":["S1","X2"],
    "X2":["S2","X1"],
    "X1":["T3","V1"],
    "V1":["P3","V2"],
    "P3":["X1","S2"],
    "S2":["E1"],
    "V2":["E1"],
    "E1":["T4","P4"],
    "T4":["E2"],
    "P4":["E2"],
    "E2":["end"]
}
def ReberGrammar(n):
    rebers_set = set()
    while len(rebers_set) < n:
        s = ""
        edge_ = "B1"
        while edge_ != "end":
            s += edge_[0]
            edge_ = random.choice(graph[edge_])
        rebers_set.add(s)
    return list(rebers_set)
def make_illegal(reber_list):
    symbols = ['B', 'T', 'P', 'S', 'X', 'V', 'E']
    illegal_list = []
    for seq in reber_list:
        seq = list(seq)
        pos = random.randint(1, len(seq) - 1)
        original_char = seq[pos]
        choices = [c for c in symbols if c != original_char]
        seq[pos] = random.choice(choices)
        illegal_list.append("".join(seq))
    return illegal_list
if __name__ == '__main__':
    strings = ReberGrammar(500)
    illegal_strings = make_illegal(strings)
    with open("500_legal_ReberGrammar.txt", "w", encoding="utf-8") as f:
        for seq in strings:
            f.write(seq + "\n")
    with open("500_illegal_ReberGrammar.txt", "w", encoding="utf-8") as f:
        for seq in illegal_strings:
            f.write(seq + "\n")