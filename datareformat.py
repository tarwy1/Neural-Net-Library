file = open("epochsvstime.csv", "r")
data = file.read()
file.close()
data = data.split(",")

count = 2
for i in range(3, len(data)+37, 3):
    data.insert(i, str(count))
    count+=1


modified_data = ""
for i in range(1, len(data)+1):
    modified_data += str(data[i-1])+","
    if i%3==0:
        modified_data += "\n"
file = open("fixedcsv.csv", "w")
file.write(modified_data)
file.close()
