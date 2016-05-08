import csv

def process():
    rows = csv.reader(open("swarm_ig_test") )
    outfile = open("swarm_ig_test.mtx", 'w');
    nodeDict = {}
    indexDict = {}
    nodeIndex = 0
    edges = 0
    for columns in rows:
        src = columns[0]
        dst = columns[1]
        value = columns[2]
        edges += 1
        if not nodeDict.get(src):
            nodeDict[src] = nodeIndex
            indexDict[nodeIndex] = src
            nodeIndex += 1
        if not nodeDict.get(dst):
            nodeDict[dst] = nodeIndex
            indexDict[nodeIndex] = dst
            nodeIndex += 1
        #outfile.write(str(nodeDict.get(src)) + ' ' +  str(nodeDict.get(dst)) + ' '  + str(value) + '\n')    
    #print nodeDict
  
    #outfile.seek(0)
    #outfile.write(str(nodeIndex) + ' ' +  str(nodeIndex) + ' ' + str(rows.line_num) + '\n')
    rows = csv.reader(open("swarm_ig_test") )
    outfile.write(str(nodeIndex) + ' ' + str(nodeIndex) + ' ' + str(edges*2) + '\n')
    for col in rows:
        src = col[0]
        dst = col[1]
        value = col[2]
        outfile.write(str(nodeDict.get(src)) + ' ' +  str(nodeDict.get(dst)) + ' '  + str(value) + '\n')
        outfile.write(str(nodeDict.get(dst)) + ' ' +  str(nodeDict.get(src)) + ' '  + str(value) + '\n')
    outfile.close();

def main():
    process()

main()
