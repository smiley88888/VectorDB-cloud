import requests

base_url = "http://24.199.123.128:8000"

def test_insert():
    response = requests.get(f"{base_url}/insert", 
                            params={
                                "id": 11,
                                "user_id": 11,
                                "text": "fgreg",
                            }
    )
    print(response.json())

def test_search():
    response = requests.get(f"{base_url}/search", 
                            params={
                                "user_id": 11,
                                "text": "fgreg",
                                "limit": 5
                            }
    )
    print(response.json())

test_insert()
test_search()

