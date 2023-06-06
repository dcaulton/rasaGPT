## Installation

Create a virtual environment

```bash
python -m venv <environment name>
```

To install with the required dependencies

```bash
pip install -r requirements.txt
```


## Place your OpenAI Key on the API_key.txt file (without quotes)

## Start the Rasa core server
```bash
rasa run --cors "*"
```

## Start Rasa actions server (this contains the ChatGPT logic, it will take a few minutes to load because it downloads the HuggingFace embedding model)
```bash
rasa run actions
```

## Start HTML server
```bash
python -m server.py
```



## Credits
* `github-markdown.css` from [github-markdown-css](https://github.com/sindresorhus/github-markdown-css) project
* `showdown.min.js` from [Showdown](https://github.com/showdownjs/showdown) project
* `webchat.js` from [Rasa Webchat](https://github.com/botfront/rasa-webchat) project
