Command to build docker image

```sh
docker build -t vandog -f Dockerfile
```

When it's finished, run the server with

```sh
docker run -p5000:5000 -it --rm vandog
```

Once the docker container is running, test the API using CURL

```sh
curl "http://localhost:8000/vandog/stylize?image_url=https://images.theconversation.com/files/443350/original/file-20220131-15-1ndq1m6.jpg"
```
