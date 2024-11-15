# database.py

import sqlite3
from logging_config import logger  # Import the globally configured logger
from datetime import datetime

def create_database(db_path='vehicle_plates.db'):
    try:
        logger.info("Creating database ....")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS vehicle_plates (
                        id INTEGER PRIMARY KEY,
                        vehicle_id INTEGER,
                        plate_number TEXT,
                        timestamp TEXT,
                        speed REAL,
                        image_path TEXT
                    )''')
        conn.commit()
        logger.info('Database and table created succesfully.')
    except Exception as e:
        logger.error(f'Error while creating database{e}',exc_info=True)
        raise
    finally:
        conn.close()

def insert_number_plate(db_path, vehicle_id, plate_number, speed, image_path):
    try:
        logger.info('Inserting values to the database that was created previously...')
        vehicle_id = int(vehicle_id)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        c.execute('''SELECT * FROM vehicle_plates WHERE vehicle_id = ?''', (vehicle_id,))
        existing_record = c.fetchone()

        if existing_record:
            c.execute('''UPDATE vehicle_plates 
                        SET speed = ?, timestamp = ?, plate_number = ?, image_path = ? 
                        WHERE vehicle_id = ?''', 
                        (speed, timestamp, plate_number, image_path, vehicle_id))
            logger.info(f"values added --> speed, timestamp, plate_number, image_path, vehicle_id")
        else:
            c.execute('''INSERT INTO vehicle_plates (vehicle_id, plate_number, timestamp, speed, image_path)
                        VALUES (?, ?, ?, ?, ?)''', 
                        (vehicle_id, plate_number, timestamp, speed, image_path))
            logger.info('Values added vehicle_id, plate_number, timestamp, speed, image_path')

         # Commit the changes to the database
        conn.commit()
    except sqlite3.DatabaseError as e:
        logger.error(f"Database error while inserting/updating vehicles info: {e}",exc_info=True)
        raise
    except Exception as e:
        logger.error(f'Unexpected error in insert_numbser_plate {e}',exc_info=True)
        raise
    finally:
        if conn:
            conn.close()

def fetch_all_data(db_path='vehicle_plates.db'):
    try:
        logger.info('Fecting all the data form the database')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM vehicle_plates")
        rows = c.fetchall()
        for row in rows:
            print(row)
    except sqlite3.Error as e:
        logger.error(f'Error while fetching data : {e}',exc_info=True)
        raise
    finally:
        conn.close()
