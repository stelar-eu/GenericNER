# Docker Instructions

To build the image run:
```
docker build -t ner .
```

To create the container run:
```
docker run -v <path-to-local-directory>/:/app/logs/ ner  template_input.json template_output.json
```

Please examine the `template_input.json` to understand the required parameters.
