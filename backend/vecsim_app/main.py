import uvicorn
from pathlib import Path
from aredis_om import (
    get_redis_connection,
    Migrator
)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from vecsim_app import config
from vecsim_app.api import routes


app = FastAPI(
    title=config.PROJECT_NAME,
    docs_url=config.API_DOCS,
    openapi_url=config.OPENAPI_DOCS
)

app.add_middleware(
        CORSMiddleware,
        allow_origins="*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
)

# Routers
app.include_router(
    routes.paper_router,
    prefix=config.API_V1_STR + "/paper",
    tags=["papers"]
)


# @app.on_event("startup")
# async def startup():
    # You can set the Redis OM URL using the REDIS_OM_URL environment
    # variable, or by manually creating the connection using your model's
    # Meta object.
    # Paper.Meta.database = get_redis_connection(url=config.REDIS_URL, decode_responses=True)
    # await Migrator().run()

# static image files
# app.mount("/data", StaticFiles(directory="data"), name="data")

if __name__ == "__main__":
    import os
    env = os.environ.get("DEPLOYMENT", "prod")

    server_attr = {
        "host": "0.0.0.0",
        "reload": True,
        "port": 8000,
        "workers": 1
    }
    if env == "prod":
        server_attr.update({"reload": False,
                            "workers": 2,
                            "ssl_keyfile": "key.pem",
                            "ssl_certfile": "full.pem"})

    uvicorn.run("main:app", **server_attr)
