import os
import json
cnt=0
for ele in os.listdir("data/Omnigibson_sg_JSON/test"):
    jsfile=os.path.join("data/Omnigibson_sg_JSON/test",ele)
    with open(jsfile,"r")as f:
        data=json.load(f)
    # if len(data)>=7:
    #     cnt+=1
    cnt+=len(data)



    