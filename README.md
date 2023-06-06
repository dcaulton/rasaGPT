## installation

To install with the required dependencies

```bash
pip install -r requirements.txt
```


## Place your OpenAI Key on the API_key.txt file (without quotes)

## Start the Rasa core server
```bash
rasa run --cors "*"
```

`xxx` is the API key we can get from https://openweathermap.org/

## Start Rasa actions server (this contains the ChatGPT logic)
```bash
rasa run --cors "*"
```

## Start HTML server
```bash
python -m server.py
```



## Credits
* `github-markdown.css` from [github-markdown-css](https://github.com/sindresorhus/github-markdown-css) project
* `showdown.min.js` from [Showdown](https://github.com/showdownjs/showdown) project
* `webchat.js` from [Rasa Webchat](https://github.com/botfront/rasa-webchat) project
