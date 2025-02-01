from fastapi import FastAPI
import os
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
import whisper
import pandas as pd
from fastapi.responses import PlainTextResponse


app = FastAPI()

api_key = 'your-openai-api-key'

def formatfile(fpath):
    try:
        os.system('npx prettier@3.4.2 --write '+'C:'+ fpath)
        return 1
    except:
        return 0 

def countdays(inputfile,outputfile,day):
    with open('C:'+inputfile, 'r') as file:
        content = file.read() 
    data = {
    "model": "gpt-4o-mini",  
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count and tell  only the number of occurrences of "+ day +"  as a number with no extra text in the following data given :   " + content }
    ]}
    header = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
   }
    response = requests.post('https://aiproxy.sanand.workers.dev/openai/v1/chat/completions', headers=header, json=data)
    if response.status_code == 200:
        result = response.json()
        x= result['choices'][0]['message']['content']
        # wriite to output day
        with open('C:'+ outputfile, 'w') as file:
            file.write(x)
        return 1
    else:
        return 0

def sorting(inputfile,outputfile):
    with open('C:'+inputfile, 'r') as file:
        content = file.read() 
    data = {
    "model": "gpt-4o-mini",  
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Sort the array of contacts in the given content by last_name, then first_name ,and give only the sorted array with no extra text  where the content is : " + content }
    ]}
    header = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
   }
    response = requests.post("https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", headers=header, json=data)
    if response.status_code == 200:
        result = response.json()
        x= result['choices'][0]['message']['content']
        with open('C:'+ outputfile, 'w') as file:
            file.write(x)
        return 1
    else:
        return 0
    
def logs(fpath,dest,count):
    folder = 'C:' + fpath
    files = os.listdir(folder)
    fileTime = []
    for file in files:
        totalp = os.path.join(folder, file)
        time = os.path.getmtime(totalp)
        fileTime.append((file, time))

    sorted_files = sorted(fileTime, key=lambda x: x[1], reverse=True)
    for i in range(0,count):
        with open(sorted_files[i][0], 'r') as file:
            x = file.readline() 
        with open('C:'+ dest, 'w') as file:
            file.write(x)
    return 1

def email(inputfile,outputfile):
    with open('C:'+inputfile, 'r') as file:
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
        # wriite to output day
        with open('C:'+ outputfile, 'w') as file:
            file.write(x)
        return 1
    else:
        return 0
    
def photo(inputfilepath,datax,outputfilepath):
    with open('C:'+inputfilepath, 'rb') as image_file:
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
                    "text": "Extract the " + datax +"from this image and give only that with no extra text"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "detail": "low",
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
        with open('C:'+ outputfilepath, 'w') as file:
            file.write(datav)
        return 1
    else:
        print('error')
        return 0

def similar(inputfile,outputfile):
    with open('C:'+inputfile, 'r') as file:
        content = file.read()
    content=list(content)
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
    if response.status_code != 200 :
        return 0
    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]

    def cossim(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    pair = None
    highest = -1
    for (i, j) in combinations(range(len(content)), 2):
        sim = cossim(embeddings[i], embeddings[j])
        if sim > highest:
            highest = sim
            pair = (i, j)

    with open('C:'+ outputfile, 'w') as file:
            file.write(content[pair[0]])
            file.write(content[pair[1]])

    return 1

def markd(inputpath,outputpath):
    pp = 'C:' + inputpath
    ans = subprocess.run(['cmd', '/c', 'dir', '/S', '/B', pp], capture_output=True, text=True)
    names= ans.stdout.splitlines()
    fullpaths= []
    for root, _, files in os.walk(pp):
        for file in files:
            fullpaths.append(os.path.join(root, file))
    headings=[]
    for paths in fullpaths:
        try:
            with open(paths, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.startswith("# "): 
                        headings.append(line.strip())
        except:
            return 0
    dataf = dict(zip(names, headings))
    vbj = json.dumps(dataf)
    with open('C:'+ outputpath, 'w') as file:
            file.write(vbj)
            return 1

def sqltask(inputfilepath,outputfilepath):
    path = "C:" + inputfilepath

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

    with open('C:'+ outputfilepath, 'w') as file:
            file.write(ans)
    return 1

def uvtask(email):
    try:
        import uvicorn
        print("uvicorn is already installed.")
    except ImportError:
        print("uvicorn not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn"])
    url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py"
    subprocess.run(["python", "-c", f"import urllib.request; exec(urllib.request.urlopen('{url}').read())"], check=True)

    subprocess.run(["uv", "run", "python", "datagen.py", email])

def ethicalpath():
    print('This path cannot be accessed ')
    return 1
def unethical_task():
    print('Data deletion and  data exfiltration is not encouraged')

def gitt(url, message, fileedit, content, targetdir):
    try:
        originalD = os.getcwd()
        os.chdir(targetdir) 
        subprocess.run(["git", "clone", url], check=True)
        repo_name = url.split("/")[-1].replace(".git", "")
        os.chdir(repo_name)
        with open(fileedit, "w") as file:
            file.write(content)
            subprocess.run(["git", "add", fileedit], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            subprocess.run(["git", "push"], check=True)
        os.chdir(originalD)
        return 1
    except:
        return 0

def fetchapi(url,outputfile):
    x=requests.get(url)
    if outputfile:
        with open('C:'+ outputfile, 'w') as file:
            file.write(x)
    return 1

def dbSqlDuck(query,dbpath,outputpath):
    dbpath='C:'+dbpath
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

    if outputpath:
        with open('C:'+ outputpath, 'w') as file:
                file.write(ans)

    return 1

def compressimage(url,outputpath):
    fp='C:'+url
    op='C:' + outputpath
    image = Image.open(fp)
    format = image.format 
    if format in ["JPEG", "JPG"] and image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if format == "WEBP":
        image.save(op, format=format, lossless=True) 
    else:
        image.save(op, format=format, optimize=True)

    return 1


def transcribe(audiopath,outputpath,lang):
    fp='C:'+audiopath
    model = whisper.load_model("medium")  
    if lang:
        lang=lang
    else:
        lang='English'
    result = model.transcribe(fp, language=lang) 
    if outputpath :
        with open('C:'+ outputpath, 'w') as file:
                file.write(result["text"])
    else:
        print(result["text"])
    return 1

def markedownHtml(filepath,outputpath):
    with open('C:'+filepath, 'r') as file:
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
        with open('C:'+ outputpath, 'w') as file:
            file.write(x)
        return 1
    else:
        return 0

def filterCSV(filtersdict,filepath,outputpath):
    try:
        df = pd.read_csv('C:'+filepath)
        filters=dict(filtersdict)
        for key, value in filters.items():
            if key in df.columns:
                df = df[df[key].astype(str) == value]
        result = df.to_dict(orient="records")
        if outputpath:
            with open('C:'+ outputpath, 'w') as file:
                file.write(json.dumps(result))
        else:
            print(json.dumps(result))
        return 1
    except:
        return 0

def scraper(url,outputpath):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

    if outputpath:
        with open('C:'+ outputpath, 'w') as file:
            file.write(data)
            return 1
    else:
        print(data)
        return 1

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
                "days": {"type": "string", "description": "Name of the day"},
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
                "inputfile": {"type": "string", "description": "Folderpath of all markedown files"},
                "outputfile": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["inputfile", "outputfile"],
        },
    },
    {
        "name": "sqltask",
        "description": "Execute a query on a database.",
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
        "description": " for creating data files on which tasks will be run",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "the email for running the data files creation"},
                    },
            "required": ["inputfilepath", "outputfilepath"],
        },
    },
    {
        "name": "ethicalpath",
        "description": "To check if inputfile includes something other than /data",
        "parameters": {
            "type": "object",
            "properties": {},
            
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
            "required": ["url"],
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
            "required": ["url",],
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
            "required": ["quer","dbpath"],
        },
    },
    {
        "name": "compressimage",
        "description": "To compress an image ",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "input file path for the image"},
                "outputpath": {"type": "string", "description": "filepath of the compressed image "},
            },
            "required": ["url"],
        },
    },
    {
        "name": "transcribe",
        "description": "Transcribe an audio file",
        "parameters": {
            "type": "object",
            "properties": {
                "audiopath": {"type": "string", "description": "Filepath of audio file"},
                "lang": {"type": "string", "description": "the languange in which audio is transcribed"},
                "outputpath": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["audiopath"],
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
        "description": "to filter a csv file",
        "parameters": {
            "type": "object",
            "properties": {
                "filtersdict": {"type": "dictionary", "description": "Filters as a dictionary of keys being filter name and values as  value pair"},
                "filepath": {"type": "string", "description": "Filepath of csv file"},
                "outputfilepath": {"type": "string", "description": "filepath where result is to be strored "},
            },
            "required": ["filepath", "filtersdict"],
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
            "required": ["filepath", "outputpath"],
        },
    }
]

@app.get("/read") #/read?path=<file path>
def read(path: str):
    try:
        with open('C:'+path, "r") as file:
            content = file.read()
        return PlainTextResponse(content) 
    except:
        return {"error": "Not Found!"},  404


@app.post("/run")
def run_task(task: str): #/run?task=<task description>
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
    if "choices" in result and result["choices"]:
        function_call = result["choices"][0]["message"].get("function_call")
        if function_call:
            name = function_call.get("name")
            arguments = function_call.get("arguments")
            funv= {
                "name": name,
                "arguments": arguments
            }
    result = globals()[funv["name"]](**funv["arguments"])
    if result == 1 :
        return 200 ,'Ok'
    else :
        return {"error": "Something went wrong!"}, 500
