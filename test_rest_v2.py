from edon import EdonClient
import math, random, pprint

client = EdonClient(base_url="http://127.0.0.1:8002")

window = {
    "physio": {
        "EDA": [0.25 for _ in range(240)],
        "BVP": [math.sin(i/6) for i in range(240)],
    },
    "motion": {
        "ACC_x": [random.gauss(0, 1) for _ in range(240)],
        "ACC_y": [random.gauss(0, 1) for _ in range(240)],
        "ACC_z": [random.gauss(0, 1) for _ in range(240)],
    },
    "task": {
        "id": "walking",
        "difficulty": 0.5,
    },
}

res = client.cav_batch_v2(
    windows=[window],
    device_profile="humanoid_full"
)

pprint.pp(res)
