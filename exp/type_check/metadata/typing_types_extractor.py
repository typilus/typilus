#!/usr/bin/env python3
from bs4 import BeautifulSoup
import requests


def main():
    url = "https://docs.python.org/3/library/typing.html"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    for dt in soup.find_all("dt"):
        mem = dt["id"].split(".")[1]
        if mem[0].isupper():
            print(mem)


if __name__ == "__main__":
    main()
