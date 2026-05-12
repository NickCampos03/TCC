import requests
import random
import time

home_url = "http://192.168.86.249:8080/"

session = requests.Session()

while True:
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # 🔹 Requisição GET (simula navegação)
        r = session.get(home_url, headers=headers)
        print("Requisição legítima enviada:", r.status_code)

    except Exception as e:
        print("Erro:", e)

    # ⏱ Tempo aleatório (simula comportamento humano)
    time.sleep(random.uniform(1, 5))
