import os 
import requests



pdf_path = "knowledge_base/artificial_intelligence_technology.pdf"
def download_pdf(pdf_path):
  os.makedirs("knowledge_base",exist_ok=True)
      #Enter the url of the pdf 
  url = "https://link.springer.com/content/pdf/10.1007/978-981-19-2879-6.pdf"
  
    
      # Local filename to save the downloaded file 
  file_name = pdf_path
  
      # Send a GET request to the url 
  response = requests.get(url , timeout=5)
  
      # check if the request was successful
  if response.status_code == 200:
        # Open the file and save it 
    with open (file_name,"wb") as  file:
        file.write(response.content)
    print(f"[Info] the file has been downloaded and saved as {file_name}")
  
  else:
    print(f"[Info] Failed to download the file. Status code: {response.status_code}")

def prepare_knowledge_base(pdf_path):
  if not os.path.exists(pdf_path):
      print(f"[Info] file doesn't exist, downloading... ")
      download_pdf(pdf_path)
  else:
     print(f"File {pdf_path} Exists.")

prepare_knowledge_base(pdf_path)