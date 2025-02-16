# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "requests",
#     "numpy",
#     "pandas",
#     "pillow",  # PIL
#     "duckdb",
#     "urllib3",     
#     "mdformat",
# ]
# ///

from fastapi import FastAPI
import os
import mdformat
import requests
from datetime import datetime
from itertools import combinations
import base64
import numpy as np
import subprocess
import json 
import sqlite3
import sys
import duckdb
from PIL import Image
import pandas as pd
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import urllib.request
import threading
import re
import asyncio
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET","POST"],  
    allow_headers=["*"],  
)

chk = False
dx={}
api_key = os.getenv("AIPROXY_TOKEN")


def countdays(inputfile,outputfile,day):
    weekday = day.lower()  
    count = 0
    with open(inputfile, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            try:
                dt = None
                if re.match(r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}", line): 
                    dt = datetime.strptime(line, "%Y/%m/%d %H:%M:%S")
                elif re.match(r"\d{4}-\d{2}-\d{2}", line): 
                    dt = datetime.strptime(line, "%Y-%m-%d")
                elif re.match(r"\d{2}-[A-Za-z]{3}-\d{4}", line): 
                    dt = datetime.strptime(line, "%d-%b-%Y")
                elif re.match(r"[A-Za-z]{3} \d{2}, \d{4}", line): 
                    dt = datetime.strptime(line, "%b %d, %Y")

                if dt and dt.strftime("%A").lower() == weekday:
                    count += 1
                else:
                    continue
            except ValueError:
                print('hu')
                continue  
    try:
        with open(outputfile, 'w') as file:
            print('yessssss')
            file.write(str(count))
            print('huonuio')
            return 1
    except:
        return 0

def sorting(inputfile,outputfile):
    print('here456')
    with open(inputfile, "r", encoding="utf-8") as file:
        contacts = json.load(file)  
        contacts.sort(key=lambda c: (c["last_name"], c["first_name"]))   
    try:
        print('helloyyy')
        with open(outputfile, "w") as file:
            json.dump(contacts, file)
            return 1
    except:
        print('donot')
        return 0
    
def logs(fpath,dest,count):
    folder = fpath
    files = os.listdir(folder)
    fileTime = []
    for file in files:
        totalp = os.path.join(folder, file)
        time = os.path.getmtime(totalp)
        fileTime.append((file, time))

    sorted_files = sorted(fileTime, key=lambda x: x[1], reverse=True)
    print(sorted_files)
    try:
        for i in range(0,count):
            with open(fpath+sorted_files[i][0], 'r') as file:
                print('success')
                x = file.readline() 
            with open(dest, 'a') as file:
                file.write(str(x))
        return 1
    except:
        return 0 

def email(inputfile,outputfile):
    with open(inputfile, 'r') as file:
        content = file.read() 
    data = {
    "model": "gpt-4o-mini",  
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Extract the senders email address and give only that with no extra text from the content given as :   " + content }
    ]}
    header = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
   }
    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=header, json=data)
    if response.status_code == 200:
        result = response.json()
        x= result['choices'][0]['message']['content']
       
        with open(outputfile, 'w') as file:
            file.write(x)
        return 1
    else:
        return 0
    
def photo(inputfilepath,datax,outputfilepath):
    sensitive_terms = ["credit card number", "bank account number", "passport number"]
    if any(term in datax.lower() for term in sensitive_terms):
        datax='long numeric sequence'
    with open(inputfilepath, 'rb') as image_file:
        image_data = image_file.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    data = {
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "please find the" + datax + " present in the image and tell just the sequence with no other text"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "detail": "high",
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }]}]}
    
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"}

    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        response_json = response.json()
        datav=response_json["choices"][0]["message"]["content"]
        datav = datav.replace(" ", "")
        with open(outputfilepath, 'w') as file:
            file.write(datav)
        return 1
    else:
        print('error')
        return 0

def similar(inputfile,outputfile):
    with open(inputfile, 'r') as file:
        content = file.read()
    content = content.split("\n")
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    response = requests.post(
        url,
        headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"},
        json={
            "input": content,
            "model": "text-embedding-3-small",
            "encoding_format": "float"
        }
    )
    print('similarchk1')
    if response.status_code != 200 :
        return 0
    print('similarchk2')
    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]

    def cossim(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    pair = None
    highest = -1
    print('similarchk13')
    for (i, j) in combinations(range(len(content)), 2):
        sim = cossim(embeddings[i], embeddings[j])
        if sim > highest:
            highest = sim
            pair = (i, j)
    print('similarchk4')
    try:
        with open(outputfile, 'a') as file:
                file.write(content[pair[0]])
                file.write('\n')
                file.write(content[pair[1]])
        print('similarchk5')
        return 1
    except:
        return 0

def markd(inputpath,outputpath):
    pp = inputpath
    fullpaths= []
    for root, _, files in os.walk(pp):
        for file in files:
            fullpaths.append(os.path.join(root, file))

    print(fullpaths)
    formatted_paths = [path.replace("\\", "/") for path in fullpaths]
    names = ["/".join(path.split("/")[-2:]) for path in formatted_paths]
    headings=[]
    for paths in fullpaths:
        try:
            with open(paths, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.startswith("# "): 
                        print(line.strip())
                        headings.append(line[2:].strip())
        except:
            return 0
    dataf = dict(zip(names, headings))
    vbj = json.dumps(dataf)
    with open(outputpath, 'w') as file:
            file.write(vbj)
            return 1

def sqltask(inputfilepath,outputfilepath):
    path = inputfilepath

    query = """
    SELECT SUM(units * price) AS total_sales
    FROM tickets
    WHERE LOWER(TRIM(type)) = 'gold';
    """
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    ans = result[0] 
    conn.close()

    try:
        with open(outputfilepath, 'w') as file:
            file.write(str(ans))
        return 1
    except:
        return 0

def ethicalpath():
    print('This path cannot be accessed ')
    return 2
def unethical_task():
    print('Data deletion and  data exfiltration is not encouraged')
    return 3

def gitt(url, message, fileedit, content, targetdir):
    try:
        originalD = os.getcwd()
        if targetdir:
            os.makedirs(targetdir, exist_ok=True)
        else:
            os.chdir('/data')
        subprocess.run(["git", "clone", url], check=True)
        repo_name = url.split("/")[-1].replace(".git", "")
        os.chdir(repo_name)
        print('hereghjkl')
        if fileedit:
            with open(fileedit, "a") as file:
                if content:
                    file.write(content)
                print('htgh')
                subprocess.run(["git", "add", fileedit], check=True)
                subprocess.run(["git", "commit", "--allow-empty","-m", message], check=True)
                subprocess.run(["git", "push"], check=True)
                print('htio')
        else:
            subprocess.run(["git", "commit", "--allow-empty","-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
        os.chdir(originalD)
        return 1
    except:
        return 0

def fetchapi(url,outputfile):
    x=requests.get(url).json()
    global chk
    global dx
    if chk:
        dx["result"] = x
        return 4
    elif outputfile:
        with open(outputfile, 'w') as file:
            file.write(str(x))
        return 1
    else:
        return 0
    

def dbSqlDuck(query,dbpath,outputpath):
    global chk
    global dx
    dbpath=dbpath
    ext = os.path.splitext(dbpath)[1]
    if ext =='duckdb':
        conn = duckdb.connect(dbpath)
        ans = conn.execute(query).fetchall()
        conn.close()
    else:
        conn = sqlite3.connect(dbpath)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        ans = result[0] 
        conn.close()

    try:
        if chk:
            dx["result"]=str(ans)
            return 4
        elif outputpath:
            with open(outputpath, 'w') as file:
                    file.write(str(ans))
            return 1
    except:
        return 0 

def compressimage(url,outputpath):
    try:
        fp=url
        op=outputpath
        image = Image.open(fp)
        format = image.format 
        if format in ["JPEG", "JPG"] and image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        if format == "WEBP":
            image.save(op, format=format, lossless=True) 
        else:
            image.save(op, format=format, optimize=True)

        return 1
    except:
        return 0


def transcribe(audiopath,outputpath,lang):
    try:
        global chk
        global dx
        fp=audiopath
        print('jhjhjh')
        if lang :
            lang=lang
        else:
            lang='en'
        result = " "
        if chk:
            dx["result"] = result
            return 4
        elif outputpath :
            with open(outputpath, 'w') as file:
                    file.write(str(result))
            return 1
        else:
            print('oh ho')
            return 0
    except:
        return 0

def markedownHtml(filepath,outputpath):
    global chk
    global dx
    with open(filepath, 'r') as file:
        content = file.read() 
    data = {
    "model": "gpt-4o-mini",  
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Convert the given markedown in the content into html and provide only the html with no extra text as response,  where the content is : " + content }
    ]}

    header = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
   }
    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=header, json=data)
    if response.status_code == 200:
        result = response.json()
        x= result['choices'][0]['message']['content']
        if chk:
            dx["result"]= x
            return 4
        elif outputpath:
            with open(outputpath, 'w') as file:
                file.write(x)
            return 1
    else:
        return 0

def filterCSV(filtersdict,filepath,outputpath):
    try:
        global chk
        global dx
        df = pd.read_csv(filepath)
        filters=dict(filtersdict)
        for key, value in filters.items():
            if key in df.columns:
                df = df[df[key].astype(str) == value]
        result = df.to_dict(orient="records")
        if chk:
            dx["result"]= json.dumps(result)
            return 4
        elif outputpath:
            with open(outputpath, 'w') as file:
                file.write(json.dumps(result))
            return 1
        else:
            return 0
    except:
        return 0

def scraper(url,outputpath):
    global chk
    global dx
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
        if chk:
            dx["result"] = data
            return 4
        elif outputpath:
            with open(outputpath, 'w') as file:
                file.write(data)
                return 1
        else:
            return 0
    except:
        return 0 

FUNCTIONS = [
    {
        "name": "formatfile",
        "description": "Get the file path for prettier formatting",
        "parameters": {
            "type": "object",
            "properties": {
                "fpath": {"type": "string", "description": "file path"}
            },
            "required": ["fpath"],
        },
    },
    {
        "name": "countdays",
        "description": "Counting Days present in a filepath ",
        "parameters": {
            "type": "object",
            "properties": {
                "inputfile": {"type": "string", "description": "input file from where dates are read"},
                "outputfile": {"type": "string", "description": "file to write the output"},
                "day": {"type": "string", "description": "Name of the day"},
            },
            "required": ["inputfile", "day", "outputfile"],
        },
    },
    {
        "name": "sorting",
        "description": "To sort the array of contacts ",
        "parameters": {
            "type": "object",
            "properties": {
                "inputfile": {"type": "string", "description": "file from where the array is read"},
                "outputfile": {"type": "string", "description": "file path where output is to be written"}
            },
            "required": ["inputfile","outputfile"],
        },
    },
    {
        "name": "logs",
        "description": "tp find the most recent log files.",
        "parameters": {
            "type": "object",
            "properties": {
                "fpath": {"type": "string", "description": "folderpath from where the log files are read"},
                "dest": {"type": "string", "description": "filepath where the output to be written"},
                "count" : {"type": "integer", "description": "number of log files"}
            },
            "required": ["fpath", "dest","count"],
        },
    },
    {
        "name": "email",
        "description": "Finding email address from emails.",
        "parameters": {
            "type": "object",
            "properties": {
                "inputfile": {"type": "string", "description": "Filepath from where email is read"},
                "outputfile": {"type": "string", "description": "filepath where result is to be stored "},
            },
            "required": ["inputfile", "outputfile"],
        },
    },
    {
        "name": "photo",
        "description": "Extracting data from image.",
        "parameters": {
            "type": "object",
            "properties": {
                "inputfilepath": {"type": "string", "description": "Filepath from where image is read"},
                "outputfilepath": {"type": "string", "description": "filepath where result is to be stored "},
                "datax": {"type": "string", "description": "The data that needs to be extracted from the image"},
            },
            "required": ["inputfilepath", "outputfilepath", "datax"],
        },
    },
    {
        "name": "similar",
        "description": "Finding similar content from a file .",
        "parameters": {
            "type": "object",
            "properties": {
                "inputfile": {"type": "string", "description": "Filepath for similar content is read"},
                "outputfile": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["inputfile", "outputfile"],
        },
    },
    {
        "name": "markd",
        "description": "Finding headings and making json of markedown file names and heading names ",
        "parameters": {
            "type": "object",
            "properties": {
                "inputpath": {"type": "string", "description": "Folderpath of all markedown files"},
                "outputpath": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["inputpath", "outputpath"],
        },
    },
    {
        "name": "sqltask",
        "description": "to find number of gold tickets ",
        "parameters": {
            "type": "object",
            "properties": {
                "inputfilepath": {"type": "string", "description": "Filepath of database file"},
                "outputfilepath": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["inputfilepath", "outputfilepath"],
        },
    },
    {
        "name": "uvtask",
        "description": " to get email for running datagen thriugh uv ",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "the script url "},
                "email": {"type": "string", "description": "the email for running the data files creation"},
                    },
            "required": ["inputfilepath", "outputfilepath"],
        },
    },
    {
        "name": "unethical_task",
        "description": "To check if task asks for data deletion or data exfiltration",
        "parameters": {
            "type": "object",
            "properties": {},
            
        },
    },
    {
        "name": "gitt",
        "description": " to clone and commit a github repo",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "the github url to commit "},
                "message": {"type": "string", "description": "the commit message"},
                "fileedit": {"type": "string", "description": "the files to edit"},
                "content": {"type": "string", "description": "the content to add in those files"},
                "targetdir": {"type": "string", "description": "the directory path where the repo is made"},
                
                    },
            "required": ["url","message","fileedit","content","targetdir"],
        },
    },
    {
        "name": "fetchapi",
        "description": "to fetch data from an api",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "the api url"},
                "outputfile": {"type": "string", "description": "the output file path where the result would be stored"},
                    
                },
            "required": ["url","outputfile"]
        },
    },
    {
        "name": "dbSqlDuck",
        "description": "Execute a query on a sql or duck database.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the query to execute"},
                "dbpath": {"type": "string", "description": "Filepath of database file"},
                "outputpath": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["query","dbpath","outputpath"],
        },
    },
    {
        "name": "compressimage",
        "description": "To compress an imagefile or imageurl and save as an compressed image output ",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "input file path for the image"},
                "outputpath": {"type": "string", "description": "filepath of the compressed image "},
            },
            "required": ["url","outputpath"],
        },
    },
    {
        "name": "transcribe",
        "description": "Transcribe an audio file",
        "parameters": {
            "type": "object",
            "properties": {
                "audiopath": {"type": "string", "description": "Filepath of audio file"},
                "lang": {"type": "string", "description": "the languange in which audio is transcribed suitable to whisper transcriber"},
                "outputpath": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["audiopath","lang","outputpath"],
        },
    },
    {
        "name": "markedownHtml",
        "description": "To convert marked down to Html.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Filepath of marked down file"},
                "outputpath": {"type": "string", "description": "filepath where html result is to be strored "},
            },
            "required": ["filepath", "outputpath"],
        },
    },
    {
        "name": "filterCSV",
        "description": "to filter a csv file with filter values given",
        "parameters": {
            "type": "object",
            "properties": {
                "filtersdict": {"type": "object", "description": "Filters as a dictionary of keys being filter name and values as  value pair"},
                "filepath": {"type": "string", "description": "Filepath of csv file"},
                "outputfilepath": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["filepath", "filtersdict","outputpath"],
        },
    },
     {
        "name": "scraper",
        "description": "To scrape a website ",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "website url"},
                "outputpath": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["url", "outputpath"],
        },
    }
]

def presence(task):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-mini",  
        "messages": [
            {"role": "system", "content": "Determine whether the given task explicitly asks for returning the result in the response body instead of saving in an outputfilepath . Reply with 'Yes' or 'No' only ."},
            {"role": "user", "content": task},
        ],
    }
    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",headers=headers,data=json.dumps(data))
    response_json = response.json()
    print(response.json())
    if "choices" in response_json and len(response_json["choices"]) > 0:
        reply = response_json["choices"][0]["message"]["content"]
        bv=reply.strip()  
    if 'Yes' in bv :
        return True
    else:
        return False

@app.get("/read") 
def read(path: str):
    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        return PlainTextResponse(content) 
    except:
        return 404


@app.post("/run")
def run_task(task: str): 
    print(api_key)
    print(task)
    global chk
    global dx
    chk=presence(task)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": task},
        ],
        "functions": FUNCTIONS,
        "function_call": "auto",
    }
    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=headers, data=json.dumps(data))
    print('here')
    print(response.status_code)
    result = response.json()
    print('here2')
    if "choices" in result and result["choices"]:
        print(result)
        function_call = result["choices"][0]["message"].get("function_call")
        if function_call:
            name = function_call.get("name")
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)  
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format for arguments")
            funv= {
                "name": name,
                "arguments": arguments
            }
    else :
        print('hi')
        print(result)
    file_keys = {"inputfile", "inputfilepath", "inputpath", "dbpath", "filepath", "fpath","url"}
    if file_keys & funv["arguments"].keys():
        op=file_keys.intersection(funv["arguments"].keys())
        dojh=next(iter(op))
        hj=funv["arguments"][dojh][:5]
        print(hj)
        if dojh != 'url' and hj !='/data':
            result = ethicalpath()
        else:
            if funv["name"] == 'uvtask':
                url=funv["arguments"]["url"]
                email=funv["arguments"]["email"]
                command = f'uv run "{url}" "{email}"'
                print(f"Executing: {command}")
                exit_code = os.system(command)
                if exit_code == 0:
                    result=1
                else:
                    result = 0
            elif funv["name"] == 'formatfile':
                fpathj = funv["arguments"]["fpath"]
                command = f'prettier --write "{fpathj}"'
                print(f"Executing: {command}")
                exit_code = os.system(command)  # Blocks execution until completion
                if exit_code == 0:
                    print("Prettier formatting successful.")
                    result=1
                else:
                    result= 0
            else:
                result = globals()[funv["name"]](**funv["arguments"])
    else:
        result = globals()[funv["name"]](**funv["arguments"])
    if chk:
        chk=False
    print('hiii')
    if result == 1 :
        print('yes')
        return 200 ,'Ok'
    elif result == 2 :
        return {"error": "Cannot access that file "}, 404
    elif result == 3 :
        return {"error": "Cannot Delete or exfilter data "}, 404
    elif result == 4:
        cv=json.dumps(dx)
        return cv , 200
    else :
        print('no')
        return {"error": "Something went wrong!"}, 500
        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)