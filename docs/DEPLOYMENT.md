# Deployment Guide

## Recommended Host

For this project, the intended free-hosting target is Render as a `Web Service`.

It should not be deployed as a static site because the app needs:
- Flask routes
- file upload handling
- server-side model training
- server-side prediction

## Included Deployment Files

- [render.yaml](../render.yaml)
- [requirements.txt](../requirements.txt)
- [../.python-version](../.python-version)

## Render Configuration

Service type:

```text
Web Service
```

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 300
```

Environment variables:

```text
PYTHON_VERSION=3.12.3
PYTHONUNBUFFERED=1
```

Health check path:

```text
/
```

## Why Python 3.12.3 Matters

During deployment, Render initially tried to use Python `3.14.3`, which caused key dependencies such as NumPy and pandas to build from source. That made installs extremely slow and unreliable on the free plan.

Pinning Python `3.12.3` allows Render to use compatible wheels and makes deployment much more predictable.

## Important Hosting Caveat

This app is demo-friendly, but not yet production-grade for free hosting:

- sessions are stored in memory
- uploaded datasets are not persisted
- model training runs synchronously in the web process
- free-tier compute can be too slow for heavier training workloads

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5001
```

## Next-Step Production Upgrades

If this project were scaled beyond a demo, the next deployment improvements would be:
- Redis or database-backed session state
- model artifact persistence
- background training jobs
- worker queue
- separate web and worker services
- batch prediction endpoints

