"Scrape recent weather data from the curiosity rover"

import json

import requests


def scrape_curiosity_weather() -> list[dict]:
    response = requests.get(
        "https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json"
    )

    if response is None:
        raise Exception("Timed out")
    if not response.ok:
        raise Exception(
            "Invalid response code: "
            + str(response.status_code)
            + ", content: \n"
            + str(response.content)
        )

    soles = json.loads(response.content)["soles"]
    soles.sort(key=lambda sol: float(sol["sol"]))

    return soles
