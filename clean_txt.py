

with open("review_3_5_2", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
        print(line)

