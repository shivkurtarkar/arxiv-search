import numpy as np
import asyncio
import redis.asyncio as redis
import config

# Delete all
async def delete_all(redis_conn):
    async for key in redis_conn.scan_iter("doc:*"):
        print(key)
        await redis_conn.delete(key)
    print("db size: ", await redis_conn.dbsize())

async def main():
    redis_conn = redis.from_url(config.REDIS_URL)
    await delete_all(redis_conn)
    print("redis db cleaned")


if __name__ == '__main__':
    asyncio.run(main())