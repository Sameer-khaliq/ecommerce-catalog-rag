# loadtest/locustfile.py
import random
from locust import HttpUser, task, between

QUERIES = [
    "wireless earbuds", "slim fit chinos", "mineral sunscreen",
    "sectional sofa", "gaming mouse", "vegan skincare",
    "action camera", "casual wear", "charcoal gray sofa",
]


class SearchUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def search(self):
        query = random.choice(QUERIES)
        self.client.post("/search", json={"query": query})