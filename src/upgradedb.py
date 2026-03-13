import sqlite3

conn = sqlite3.connect("database.db")
c = conn.cursor()

# Add new columns if not already present
columns = ["ip", "country", "region", "city", "lat", "lon"]
for col in columns:
    try:
        c.execute(f"ALTER TABLE url_logs ADD COLUMN {col} TEXT;")
    except:
        pass  # ignore if column already exists

conn.commit()
conn.close()
print("✅ Database updated with location columns!")
