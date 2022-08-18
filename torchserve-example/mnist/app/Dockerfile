# baseimage
FROM python:3.9

# install pipenv
RUN pip install pipenv

# creates directory "app" and moves there
WORKDIR /app

# copy Pipfile and Pifile.lock to current directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# create virtual environment directly on the system
RUN pipenv install --system --deploy

# copy the python script we need
COPY . ./

# expose the port
EXPOSE 9696

# define entrypoint
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app:app"]
