import requests

url = 'http://localhost:9696/predict'

customer = {
    "v1": -0.4784271960990545,
    "v2": 0.1421651242876125,
    "v3": -0.0468380289121661,
    "v4": 0.6833502680870056,
    "v5": 0.0678198572410411,
    "v6": -0.4048984594640247,
    "v7": -0.2064959660918449,
    "v8": 0.1843661072753982,
    "v9": -0.7629347043611346,
    "v10": -0.2283918022491541,
    "v11": 0.6609030198474657,
    "v12": -0.3875197498276417,
    "v13": -0.533249442461466,
    "v14": -0.502265671964952,
    "v15": 0.4051430868662198,
    "v16": -0.0606910405034945,
    "v17": -0.2072367443980979,
    "v18": 0.3056026116425647,
    "v19": 0.1348757242981795,
    "v20": -0.0339211457440372,
    "v21": 0.0989766935994606,
    "v22": -0.0751912083426939,
    "v23": -0.4814893372271562,
    "v24": 0.6788996384076292,
    "v25": -0.0115196803518529,
    "v26": 0.4090207868545027,
    "v27": 0.0758593945999199,
    "v28": -0.4471391388557191,
    "amount": 1534.53
}

try:
    response = requests.post(url, json=customer)
    response.raise_for_status()
    result = response.json()
    print(result)
    if result['fraud']:
        print('Fraud detected')
    else:
        print('No fraud')
except requests.exceptions.RequestException as e:
    print('Request error:', e)
except requests.exceptions.HTTPError as e:
    print('HTTP error:', e)
except requests.exceptions.JSONDecodeError as e:
    print('JSON decoding error:', e)
