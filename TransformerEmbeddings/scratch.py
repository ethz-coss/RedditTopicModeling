import duckdb
import config

duck_database = duckdb.connect(config.DATA_BASE_PATH)
print(duck_database.sql("SELECT created_utc FROM submissions LIMIT 10"))


