release: python manage.py migrate
web: daphne web_system.asgi: --port $PORT  --bind 0.0.0.0 -v2
celery: celery -A web_system.celery woker --pool=solo -l info 
celerybeat: celery -A web_system beat -l INFOR